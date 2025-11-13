
import os
import sys
import argparse
import hashlib
from pathlib import Path
from collections import OrderedDict  # Changed from defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Union
import struct
import socket

try:
    from scapy.all import PcapReader, PcapWriter, Packet
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import Ether

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class TrafficSplitter:
    """Main class for the traffic splitter."""

    def __init__(self):
        self.packet_count = 0

    def get_session_key(self, packet: Packet) -> Optional[str]:
        """Extracts a session identifier from a packet."""
        try:
            if IP in packet:
                layer3 = packet[IP]
                proto_l4 = layer3.proto
            elif IPv6 in packet:
                layer3 = packet[IPv6]
                proto_l4 = layer3.nh
            else:
                return None

            src_ip = layer3.src
            dst_ip = layer3.dst

            if proto_l4 == 6 and TCP in packet:  # TCP
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                protocol = "TCP"
            elif proto_l4 == 17 and UDP in packet:  # UDP
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                protocol = "UDP"
            else:
                return self._normalize_host_pair(src_ip, dst_ip)

            return self._normalize_session_key(protocol, src_ip, src_port, dst_ip, dst_port)
        except Exception:
            return None

    def _normalize_session_key(self, protocol: str, src_ip: str, src_port: int,
                               dst_ip: str, dst_port: int) -> str:
        if self._is_lower_endpoint(src_ip, src_port, dst_ip, dst_port):
            return f"{protocol}_{self._sanitize_ip(src_ip)}_{src_port}_{self._sanitize_ip(dst_ip)}_{dst_port}"
        else:
            return f"{protocol}_{self._sanitize_ip(dst_ip)}_{dst_port}_{self._sanitize_ip(src_ip)}_{src_port}"

    def _normalize_host_pair(self, src_ip: str, dst_ip: str) -> str:
        try:
            src_ip_bytes = socket.inet_pton(socket.AF_INET if '.' in src_ip else socket.AF_INET6, src_ip)
            dst_ip_bytes = socket.inet_pton(socket.AF_INET if '.' in dst_ip else socket.AF_INET6, dst_ip)
            if src_ip_bytes <= dst_ip_bytes:
                return f"HostPair_{self._sanitize_ip(src_ip)}_{self._sanitize_ip(dst_ip)}"
            else:
                return f"HostPair_{self._sanitize_ip(dst_ip)}_{self._sanitize_ip(src_ip)}"
        except socket.error:
            if src_ip <= dst_ip:
                return f"HostPair_{self._sanitize_ip(src_ip)}_{self._sanitize_ip(dst_ip)}"
            else:
                return f"HostPair_{self._sanitize_ip(dst_ip)}_{self._sanitize_ip(src_ip)}"

    def _is_lower_endpoint(self, ip1: str, port1: int, ip2: str, port2: int) -> bool:
        try:
            af1 = socket.AF_INET if '.' in ip1 else socket.AF_INET6
            af2 = socket.AF_INET if '.' in ip2 else socket.AF_INET6
            ip1_bytes = socket.inet_pton(af1, ip1)
            ip2_bytes = socket.inet_pton(af2, ip2)
            if af1 != af2: return af1 < af2
            if ip1_bytes < ip2_bytes: return True
            if ip1_bytes > ip2_bytes: return False
            return port1 <= port2
        except socket.error:
            if ip1 < ip2: return True
            if ip1 > ip2: return False
            return port1 <= port2

    def _sanitize_ip(self, ip_str: str) -> str:
        return ip_str.replace('.', '-').replace(':', '-')

    def split_pcap_file(self, input_file: str, output_dir: str,
                        parallel_sessions_limit: int = 1000) -> bool:  # Default limit for writers

        print(f"Processing file: {os.path.basename(input_file)}.")
        os.makedirs(output_dir, exist_ok=True)

        session_writers: OrderedDict[str, PcapWriter] = OrderedDict()  # Use OrderedDict for LRU
        local_packet_count = 0
        files_created_this_run = set()  # Keep track of files we attempt to create

        try:
            with PcapReader(input_file) as pcap_reader:
                for packet in pcap_reader:
                    local_packet_count += 1
                    self.packet_count += 1

                    session_key = self.get_session_key(packet)
                    if not session_key:
                        continue

                    writer = None
                    if session_key in session_writers:
                        writer = session_writers[session_key]
                        session_writers.move_to_end(session_key)  # Mark as recently used
                    else:
                        # New session, check if we need to cull an old writer
                        if len(session_writers) >= parallel_sessions_limit:
                            try:
                                # Pop the least recently used (oldest) item
                                old_key, old_writer = session_writers.popitem(last=False)
                                old_writer.close()
                                # print(f"\nClosed writer for session (LRU): {old_key} to make space.")
                            except Exception as e:
                                print(f"\nError closing LRU writer for session {old_key}: {e}")

                        base_name = Path(input_file).stem
                        output_file_path = os.path.join(output_dir, f"{base_name}.{session_key}.pcap")
                        try:
                            # Open in append mode. Scapy's PcapWriter creates if not exists, appends if exists.
                            writer = PcapWriter(output_file_path, append=True)
                            session_writers[session_key] = writer
                            files_created_this_run.add(output_file_path)
                        except Exception as e:
                            print(f"\nError creating PcapWriter for {output_file_path}: {e}")
                            # If writer creation fails, we might skip this packet or this session
                            continue

                    if writer:  # Ensure writer is valid
                        try:
                            writer.write(packet)
                        except Exception as e:
                            print(f"\nError writing packet to {getattr(writer, 'filename', 'N/A')}: {e}")

                    if local_packet_count % 1000 == 0:
                        print(
                            f"\rProcessed {local_packet_count} packets from {os.path.basename(input_file)} ({len(session_writers)} active writers)...",
                            end='', flush=True)

            print(f"\rFinished processing {local_packet_count} packets from {os.path.basename(input_file)}.")

        except Exception as e:
            print(f"\nError reading/processing {input_file}: {e}")
            return False
        finally:
            print(f"Closing {len(session_writers)} remaining session files...")
            for key, writer_instance in session_writers.items():
                try:
                    writer_instance.close()
                except Exception as e:
                    print(
                        f"Error closing writer for session {key} ({getattr(writer_instance, 'filename', 'N/A')}): {e}")

        final_files_count = 0
        # print(f"Checking {len(files_created_this_run)} potentially created files for emptiness...")
        for file_path in list(files_created_this_run):
            if os.path.exists(file_path):
                try:
                    if os.path.getsize(file_path) == 0:
                        os.remove(file_path)
                    else:
                        final_files_count += 1
                except Exception as e:
                    print(f"Error checking/removing empty file {file_path}: {e}")

        print(f"Kept {final_files_count} non-empty session files in {output_dir}")
        return True

    def remove_duplicates(self, directory: str):
        print(f"Removing duplicates in {directory}...")
        file_hashes: Dict[str, str] = {}
        duplicates_removed = 0

        for root, _, files in os.walk(directory):
            for file_name in files:
                if file_name.lower().endswith('.pcap'):
                    file_path = os.path.join(root, file_name)
                    try:
                        if not os.path.exists(file_path): continue  # Might have been removed by empty check
                        with open(file_path, 'rb') as f_handle:
                            file_hash = hashlib.md5(f_handle.read()).hexdigest()

                        if file_hash in file_hashes:
                            os.remove(file_path)
                            duplicates_removed += 1
                        else:
                            file_hashes[file_hash] = file_path
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        print(f"Error processing {file_path} for duplicate check: {e}")

        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate files from {directory}")
        else:
            print(f"No duplicate files found in {directory}")

    def process_dataset(self, root_path: str, dataset_name: str, max_writers: int):
        print(f"Processing dataset: {dataset_name}")
        print(f"Root path: {root_path}")
        self.packet_count = 0

        input_base = Path(root_path) / "1_Pcap" / dataset_name
        output_base = Path(root_path) / "2_Session" / dataset_name / "ts"

        categories = ["Malicious", "Benign"]

        for category in categories:
            input_dir = input_base / category
            output_dir_category_base = output_base / category

            if not input_dir.exists():
                print(f"Warning: Input directory not found: {input_dir}")
                continue

            print(f"\nProcessing category: {category}")

            pcap_files = list(input_dir.rglob('*.pcap'))
            print(f"Found {len(pcap_files)} PCAP files in {input_dir}")

            for i, pcap_file_path in enumerate(pcap_files, 1):
                file_stem = pcap_file_path.stem
                file_specific_output_dir = output_dir_category_base / file_stem

                print(f"\n[{i}/{len(pcap_files)}] Processing: {pcap_file_path.name} -> {file_specific_output_dir}")

                self.split_pcap_file(str(pcap_file_path), str(file_specific_output_dir),
                                     max_writers)  # Pass max_writers

            if output_dir_category_base.exists():
                print(f"\nRunning duplicate removal for category: {category} in {output_dir_category_base}")
                self.remove_duplicates(str(output_dir_category_base))

        print(f"\nDataset processing completed: {dataset_name}. Total packets processed: {self.packet_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Splitter - Python replacement for SplitCap. Splits pcap files by session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --root-path /path/to/data --dataset USTC-TFC2016-master
  %(prog)s --dataset MCFP  (uses default root-path)
  %(prog)s --single-file input.pcap --output-dir ./output_sessions
        """
    )

    main_group = parser.add_mutually_exclusive_group(required=False)
    main_group.add_argument('--dataset', type=str, default="MCFP",
                            help='Dataset name to process (e.g., USTC-TFC2016-master, MCFP)')
    main_group.add_argument('--single-file', type=str,
                            help='Process a single PCAP file')

    parser.add_argument('--root-path', type=str, default="E:\\a_TrafficDetection\\tempd_data",
                        help='Root path for dataset processing (default: current working directory)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for single file processing. If not set, defaults to <filename_stem>_sessions.')
    parser.add_argument('--max-writers', type=int, default=1000,
                        help='Maximum concurrent PcapWriter instances (default: 1000)')

    args = parser.parse_args()
    splitter = TrafficSplitter()

    if args.dataset:
        if not Path(args.root_path).exists():
            print(f"Error: Root path does not exist: {args.root_path}")
            sys.exit(1)
        # Pass max_writers to process_dataset, which then passes it to split_pcap_file
        splitter.process_dataset(args.root_path, args.dataset, args.max_writers)

    elif args.single_file:
        input_pcap = Path(args.single_file)
        if not input_pcap.exists():
            print(f"Error: Input file does not exist: {args.single_file}")
            sys.exit(1)

        output_directory = args.output_dir
        if not output_directory:
            output_directory = f"{input_pcap.stem}_sessions_py"

        splitter.split_pcap_file(str(input_pcap), output_directory, args.max_writers)  # Pass max_writers
        print(f"\nRunning duplicate removal for single file output in {output_directory}")
        splitter.remove_duplicates(output_directory)


if __name__ == "__main__":
    from time import time

    start_time = time()
    main()
    print(f"\nTotal execution time: {time() - start_time:.2f} seconds")
