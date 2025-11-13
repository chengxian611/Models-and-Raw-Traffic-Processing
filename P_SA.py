import os
import re
import dpkt
import socket
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle
from typing import Union, List, Tuple, Dict, Any, Optional
from itertools import islice
from tqdm import tqdm
import io
import tempfile
import shutil
from pathlib import Path
import argparse

# --- Configuration ---
DEFAULT_MAX_PACKET_LENGTH = 64
DEFAULT_MAX_PACKETS_PER_SESSION = 8
DATASET_NAMES = ["MCFP", "USTC-TFC2016-master", "CIRA-CIC-DoHBrw-2020"] # Add or modify as needed

# --- Helper Functions ---

def extract_five_tuple_from_packet_buffer(ip_packet: dpkt.ip.IP) -> Union[Tuple[str, str, int, int, int], None]:
    """从单个IP数据包中提取五元组 (如果已经是IP层)"""
    try:
        proto = ip_packet.p
        src_ip = socket.inet_ntoa(ip_packet.src)
        dst_ip = socket.inet_ntoa(ip_packet.dst)
        src_port = dst_port = 0 # Default for non-TCP/UDP

        if isinstance(ip_packet.data, dpkt.tcp.TCP):
            tcp = ip_packet.data
            src_port = tcp.sport
            dst_port = tcp.dport
        elif isinstance(ip_packet.data, dpkt.udp.UDP):
            udp = ip_packet.data
            src_port = udp.sport
            dst_port = udp.dport
        return (src_ip, dst_ip, src_port, dst_port, proto)
    except Exception:
        return None

def process_pcap_file_to_packet_features(
    pcap_file_path: str, 
    max_packet_length: int, 
    max_packets_per_session: int
) -> Optional[Tuple[np.ndarray, int, Optional[Tuple[str, str, int, int, int]]]]:
    """
    处理单个会话pcap文件，提取数据包级特征。
    返回: (特征矩阵, 实际数据包数量, 五元组) 或 None
    特征矩阵形状: (max_packets_per_session, max_packet_length)
    """
    packet_features_list = []
    five_tuple = None
    actual_packet_count = 0
    # root_time_stamp = 0
    # get_root_time_stamp = False
    try:
        with open(pcap_file_path, 'rb') as f:
            pcap_reader = dpkt.pcap.Reader(f)
            time_stamps = []
            for ts, buf in pcap_reader:
                if actual_packet_count >= max_packets_per_session:
                    break # 已达到每个会话的最大数据包数

                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue # 只处理IP包

                ip_packet = eth.data
                
                # 提取第一个有效IP包的五元组
                if five_tuple is None:
                    five_tuple = extract_five_tuple_from_packet_buffer(ip_packet)

                # 提取数据包的原始字节 (从IP层开始，或者整个buf如果需要链路层)
                # 这里我们用整个buf的前 N 字节，和之前session_to_numpy保持一定程度相似性
                # 但更精确的做法可能是从 ip_packet.pack() 开始
                packet_data = buf 
                
                processed_packet_vec = np.frombuffer(packet_data, dtype=np.uint8)

                if len(processed_packet_vec) > max_packet_length:
                    processed_packet_vec = processed_packet_vec[:max_packet_length]
                elif len(processed_packet_vec) < max_packet_length:
                    padding = np.zeros(max_packet_length - len(processed_packet_vec), dtype=np.uint8)
                    processed_packet_vec = np.concatenate((processed_packet_vec, padding))
                
                packet_features_list.append(processed_packet_vec)
                actual_packet_count += 1
                time_stamps.append(ts)
        
        if not packet_features_list: # pcap为空或不含有效IP包
            return None

        # 填充会话中的数据包数量
        session_feature_matrix = np.zeros((max_packets_per_session, max_packet_length), dtype=np.uint8)
        time_stamps_matrix = np.zeros((max_packets_per_session,), dtype=np.float32)
        num_packets_to_fill = min(len(packet_features_list), max_packets_per_session)
        rts = time_stamps[0]
        # time_stamps = [r - rts for r in time_stamps]
        for i in range(num_packets_to_fill):
            session_feature_matrix[i, :] = packet_features_list[i]
            time_stamps_matrix[i] = time_stamps[i]
        if  (l := max_packets_per_session - num_packets_to_fill) > 0:
            time_stamps_matrix[num_packets_to_fill:] = time_stamps[-1]
        return session_feature_matrix, num_packets_to_fill, five_tuple, time_stamps_matrix

    except dpkt.dpkt.NeedData:
        # print(f"Warning: Incomplete pcap file (NeedData): {pcap_file_path}")
        if packet_features_list: # 如果已经解析了一些包
            session_feature_matrix = np.zeros((max_packets_per_session, max_packet_length), dtype=np.uint8)
            time_stamps_matrix = np.zeros((max_packets_per_session,), dtype=np.float32)
            num_packets_to_fill = min(len(packet_features_list), max_packets_per_session)
            rts = time_stamps[0]
            time_stamps = [r - rts for r in time_stamps]
            for i in range(num_packets_to_fill):
                session_feature_matrix[i, :] = packet_features_list[i]
                time_stamps_matrix[i] = time_stamps[i]
            return session_feature_matrix, num_packets_to_fill, five_tuple, time_stamps_matrix
        return None
    except Exception as e:
        # print(f"Error processing pcap file {pcap_file_path}: {e}")
        return None


def process_one_session_task(task_args: Tuple[str, int, int, int]) -> Optional[Tuple[np.ndarray, int, int, Optional[Tuple[str, str, int, int, int]]]]:
    """
    包装器函数，用于多进程处理。
    task_args: (file_path, label_id, max_packet_length, max_packets_per_session)
    返回: (特征矩阵, 实际数据包数量, 标签ID, 五元组) 或 None
    """
    file_path, label_id, max_pkt_len, max_pkts_sess = task_args
    result = process_pcap_file_to_packet_features(file_path, max_pkt_len, max_pkts_sess)
    if result:
        features_matrix, actual_pkt_count, five_tuple, time_matrix = result
        if five_tuple is None: # 如果无法从pcap中提取五元组，则跳过
            # print(f"Warning: Could not extract five_tuple from {file_path}, skipping.")
            return None
        return features_matrix, actual_pkt_count, label_id, five_tuple, time_matrix
    return None

def get_base_app_name(app_dir_name: str) -> str:
    """从应用流量文件夹名称中提取基础应用名称"""
    app_dir_name = app_dir_name.lower()
    match = re.match(r'^[^_\-\d]{1,}', app_dir_name)
    return match.group(0) if match else app_dir_name

class PacketBatchStorage:
    """批次数据存储管理器 (针对数据包级特征)"""
    def __init__(self, save_dir: str, max_packets_per_session: int, max_packet_length: int):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_count = 0
        self.total_samples = 0
        self.max_packets_per_session = max_packets_per_session
        self.max_packet_length = max_packet_length
        
    def save_batch(self, batch_data: Dict[str, Any]) -> None:
        if not batch_data or not batch_data['session_features']:
            return
            
        batch_dir = self.save_dir / f"batch_{self.batch_count}"
        batch_dir.mkdir(exist_ok=True)
        
        # 特征是列表的NumPy数组，需要先堆叠
        session_features_np = np.array(batch_data['session_features'], dtype=np.uint8)
        np.save(batch_dir / "session_features.npy", session_features_np)
        
        actual_packet_counts_np = np.array(batch_data['actual_packet_counts'], dtype=np.int32)
        np.save(batch_dir / "actual_packet_counts.npy", actual_packet_counts_np)

        labels_np = np.array(batch_data['labels'], dtype=np.int32)
        np.save(batch_dir / "labels.npy", labels_np)
        
        with open(batch_dir / "five_tuples.pkl", 'wb') as f:
            pickle.dump(batch_data['five_tuples'], f)
        np.save(batch_dir / "time_stamps.npy", np.array(batch_data['time_stamps'], dtype=np.float32))
        self.total_samples += len(batch_data['session_features'])
        self.batch_count += 1
        
    def cleanup(self):
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)

def preprocess_sessions_to_packet_features(
    root_dir: str, 
    dataset_name: str,
    max_packet_length: int, 
    max_packets_per_session: int,
    batch_size: int = 10000 # 批处理大小可以根据内存调整
):
    label_map: Dict[str, int] = {}
    label_counter = 0
    all_tasks: List[Tuple[str, int, int, int]] = []
    
    # 创建临时目录用于存储批次数据
    temp_dir_path = Path(tempfile.mkdtemp(prefix=f"{dataset_name}_packet_data_"))
    storage = PacketBatchStorage(str(temp_dir_path), max_packets_per_session, max_packet_length)
    
    print(f"Scanning pcap files in: {root_dir}")
    # 遍历恶意和良性流量目录 (假设结构与session_to_numpy.py中的类似)
    for traffic_type in os.listdir(root_dir):
        if traffic_type not in ["Malicious", "Benign"]: # 根据实际目录结构调整
            # print(f"Skipping non-Malicious/Benign directory: {traffic_type}")
            continue
        traffic_dir = os.path.join(root_dir, traffic_type)
        if not os.path.isdir(traffic_dir):
            continue
        
        print(f"\nLoading {traffic_type}...")
        for app_name in os.listdir(traffic_dir):
            app_dir = os.path.join(traffic_dir, app_name)
            if not os.path.isdir(app_dir):
                continue
            
            pcap_files_in_app = []
            for file_name in os.listdir(app_dir):
                if file_name.endswith('.pcap'): # 或其他会话文件扩展名
                    pcap_files_in_app.append(os.path.join(app_dir, file_name))
            
            if not pcap_files_in_app:
                # print(f"No pcap files found in {app_dir}")
                continue

            print(f"\t{app_name}: {len(pcap_files_in_app)} pcap files")
            
            full_label_name = f"{traffic_type}_{get_base_app_name(app_name)}"
            if full_label_name not in label_map:
                label_map[full_label_name] = label_counter
                label_counter += 1
            
            current_label_id = label_map[full_label_name]
            for pcap_file in pcap_files_in_app:
                all_tasks.append((pcap_file, current_label_id, max_packet_length, max_packets_per_session))

    if not all_tasks:
        print("Error: No pcap files found to process.")
        storage.cleanup()
        return None
        
    print(f"\nTotal {len(all_tasks)} session files to process.")
    
    # 多进程处理
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes for parsing.")

    with Pool(processes=num_processes) as pool:
        for i in tqdm(range(0, len(all_tasks), batch_size), desc="Processing session batches"):
            batch_tasks = all_tasks[i:i+batch_size]
            results = list(tqdm(pool.imap_unordered(process_one_session_task, batch_tasks), total=len(batch_tasks), desc="  Sub-batch", leave=False))
            
            # 过滤 None 的结果并解包
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                continue

            batch_session_features, batch_actual_pkt_counts, batch_labels, batch_five_tuples, batch_time_stamps = zip(*valid_results)
            
            storage.save_batch({
                'session_features': list(batch_session_features),
                'actual_packet_counts': list(batch_actual_pkt_counts),
                'labels': list(batch_labels),
                'five_tuples': list(batch_five_tuples),
                'time_stamps': list(batch_time_stamps)
            })

    print(f"\nMerging {storage.batch_count} processed batches...")
    
    all_session_features_list = []
    all_actual_packet_counts_list = []
    all_labels_list = []
    all_five_tuples_list = []
    all_time_stamps_list = []

    for batch_id in tqdm(range(storage.batch_count), desc="Merging batches"):
        batch_dir = storage.save_dir / f"batch_{batch_id}"
        if batch_dir.exists():
            all_session_features_list.append(np.load(batch_dir / "session_features.npy"))
            all_actual_packet_counts_list.append(np.load(batch_dir / "actual_packet_counts.npy"))
            all_labels_list.append(np.load(batch_dir / "labels.npy"))
            all_time_stamps_list.append(np.load(batch_dir / "time_stamps.npy"))
            with open(batch_dir / "five_tuples.pkl", 'rb') as f:
                all_five_tuples_list.extend(pickle.load(f))
    
    if not all_session_features_list:
        print("Error: No data to merge after processing.")
        storage.cleanup()
        return None

    # 合并数据
    final_session_features = np.concatenate(all_session_features_list, axis=0)
    final_actual_packet_counts = np.concatenate(all_actual_packet_counts_list, axis=0)
    final_labels = np.concatenate(all_labels_list, axis=0)
    final_time_stamps = np.concatenate(all_time_stamps_list, axis=0)
    # final_five_tuples 已经是一个列表了

    # --- 保存最终数据 ---
    # 保存目录结构: dataset/<datasetName>/P_packet_traffic_data/
    # "P" for Packet-level features
    output_prefix = "P" 
    save_base_dir = Path("dataset") / dataset_name
    save_data_dir = save_base_dir / f"{output_prefix}_SA_traffic_data"
    save_raw_data_dir = os.path.join(save_data_dir, "raw_data")
    save_raw_data_dir = Path(save_raw_data_dir)
    save_raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving processed data to: {save_data_dir}")

    np.save(save_raw_data_dir / "vectors.npy", final_session_features)
    np.save(save_raw_data_dir / "actual_packet_counts.npy", final_actual_packet_counts)
    np.save(save_raw_data_dir / "labels.npy", final_labels)
    np.save(save_raw_data_dir / "time_stamps.npy", final_time_stamps)
    with open(save_raw_data_dir / "five_tuples.pkl", 'wb') as f:
        pickle.dump(all_five_tuples_list, f)
    
    with open(save_raw_data_dir / "label_map.pkl", 'wb') as f:
        pickle.dump(label_map, f)

    dataset_info = {
        'total_samples': len(final_session_features),
        'feature_shape': final_session_features.shape[1:], # (max_packets_per_session, max_packet_length)
        'max_packet_length': max_packet_length,
        'max_packets_per_session': max_packets_per_session,
        'data_path': str(save_data_dir),
        'label_map_path': str(save_data_dir / "label_map.pkl"),
        'five_tuples_path': str(save_data_dir / "five_tuples.pkl"),
        'time_stamps_path': str(save_data_dir / "time_stamps.npy"),
    }
    with open(save_data_dir / "dataset_info.pkl", 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"\nPacket-level feature processing complete.")
    print(f"  Total sessions processed: {dataset_info['total_samples']}")
    print(f"  Feature shape per session: {dataset_info['feature_shape']}")
    print(f"  Data saved in: {save_data_dir}")

    storage.cleanup()
    print("Temporary batch files cleaned up.")
    return dataset_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess session pcap files to packet-level features.")
    parser.add_argument("-d", "--dataset_index", type=int, default=0, choices=range(len(DATASET_NAMES)),
                        help=f"Index of the dataset to process. Choices: {list(range(len(DATASET_NAMES)))} for {DATASET_NAMES}")
    parser.add_argument("--max_pkt_len", type=int, default=DEFAULT_MAX_PACKET_LENGTH,
                        help="Maximum length (bytes) to keep for each packet.")
    parser.add_argument("--max_pkts_sess", type=int, default=DEFAULT_MAX_PACKETS_PER_SESSION,
                        help="Maximum number of packets to keep for each session.")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Number of session files to process in one go before saving to temp storage.")
    # ROOT_DIR for session pcap files, similar to session_to_numpy.py
    # This needs to point to the directory structure like: E:\a_TrafficDetection\tempd_data\2_Session\<datasetName>
    parser.add_argument("--root_dir_base", type=str, default=r"E:\a_TrafficDetection\tempd_data\2_Session",
                        help="Base root directory where dataset session pcap folders are located.")
    
    cli_args = parser.parse_args()

    selected_dataset_name = DATASET_NAMES[cli_args.dataset_index]
    
    # Construct the root directory for the selected dataset's session files
    # This assumes the session pcap files are in a structure like: <root_dir_base>/<selected_dataset_name>
    # And under that, Malicious/Benign subdirectories.
    current_root_dir = os.path.join(cli_args.root_dir_base, selected_dataset_name)

    if not os.path.isdir(current_root_dir):
        print(f"Error: Root directory for dataset '{selected_dataset_name}' not found: {current_root_dir}")
        print(f"Please ensure the directory structure is: {cli_args.root_dir_base}/{selected_dataset_name}/[Malicious|Benign]/...")
        exit(1)

    print(f"--- Starting Packet-Level Feature Preprocessing ---")
    print(f"Dataset: {selected_dataset_name}")
    print(f"Root Directory for Sessions: {current_root_dir}")
    print(f"Max Packet Length: {cli_args.max_pkt_len}")
    print(f"Max Packets per Session: {cli_args.max_pkts_sess}")
    print(f"Processing Batch Size: {cli_args.batch_size}")
    
    result_info = preprocess_sessions_to_packet_features(
        root_dir=current_root_dir,
        dataset_name=selected_dataset_name,
        max_packet_length=cli_args.max_pkt_len,
        max_packets_per_session=cli_args.max_pkts_sess,
        batch_size=cli_args.batch_size
    )

    if result_info:
        print("\n--- Preprocessing Summary ---")
        for key, value in result_info.items():
            print(f"  {key}: {value}")
    else:
        print("\n--- Preprocessing Failed ---")
