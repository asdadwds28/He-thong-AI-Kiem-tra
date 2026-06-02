import time
import numpy as np

# --- QUY TRÌNH ĐÓNG GÓI ---
# Thứ tự cam rất quan trọng để xác định vật được phép
CAM_ORDER = ["cam_1", "cam_2", "cam_3", "cam_4"]

PACKING_RULES = {
    "cam_1": {1: "Den_nho", 2: "Den_nho", 3: "Den_nho"},
    "cam_2": {4: "Den_to", 5: "Board"},
    "cam_3": {6: "rgb", 7: "day_xam", 8: "day_trang"},
    "cam_4": {9: "tui", 10: "sac"}
}

# ... (Class Slot ) ...
class Slot:
    def __init__(self, slot_id, expected_item):
        self.id = slot_id
        self.expected_item = expected_item
        self.obb_points = None
        self.center = None
        self.state = "empty" 
        self.current_item_class = None
        self.first_oke_time = None 
        self.is_saved = False 

    def update_position(self, obb_points, center):
        self.obb_points = np.array(obb_points, dtype=np.int32)
        self.center = np.array(center, dtype=np.int32)

    def set_state(self, new_state, item_class=None):
        if new_state == "oke":
            if self.state != "oke":
                self.first_oke_time = time.time()
            elif self.first_oke_time is not None:
                elapsed = time.time() - self.first_oke_time  #Tính thời gian để xác nhận 
                if elapsed >= 3.0 and not self.is_saved:
                    self.is_saved = True 
                    print(f"💾 Slot {self.id} SAVED")
        else:
            self.first_oke_time = None
        self.state = new_state
        self.current_item_class = item_class

    def reset_state(self):
        self.state = "empty"
        self.current_item_class = None
        self.first_oke_time = None
        self.is_saved = False   

class CameraConfig:
    def __init__(self, cam_name):
        self.cam_name = cam_name
        self.slots = {}
        self.cam_state = "waiting"
        self.status_message = "WAITING"
        self.has_finished_once = False 

        # Biến chứa tên vật phẩm SAI QUY TRÌNH (nếu detect thấy)
        self.forbidden_item_detected = None 

        # --- LOGIC TẠO DANH SÁCH VẬT PHẨM ĐƯỢC PHÉP (Cumulative) ---
        self.allowed_classes = set()
        if cam_name in CAM_ORDER:
            current_idx = CAM_ORDER.index(cam_name)
            # Được phép thấy vật của Cam hiện tại VÀ các Cam trước đó
            for i in range(current_idx + 1):
                c_name = CAM_ORDER[i]
                if c_name in PACKING_RULES:
                    self.allowed_classes.update(PACKING_RULES[c_name].values())
        
        # Mapping ID
        if cam_name in ["cam_1", "cam_2"]: self.id_mapping = {1:1, 2:2, 3:3, 4:4, 5:5}
        elif cam_name in ["cam_3", "cam_4"]: self.id_mapping = {1:6, 2:7, 3:8, 4:9, 5:10}
        else: self.id_mapping = {}

        if cam_name in PACKING_RULES:
            for s_id, exp_item in PACKING_RULES[cam_name].items():
                self.slots[s_id] = Slot(s_id, exp_item)

    def get_slot_by_local_id(self, local_id):
        global_id = self.id_mapping.get(local_id)
        return self.slots.get(global_id)

    def get_item_counts(self):
        # ... (Giữ nguyên logic cũ) ...
        stats = {}
        if self.cam_name in PACKING_RULES:
            req_items = list(PACKING_RULES[self.cam_name].values())
            unique_items = set(req_items)
            for item in unique_items:
                stats[item] = {"count": 0, "total": req_items.count(item), "done": False}

        all_saved = True
        for slot in self.slots.values():
            if slot.expected_item in stats:
                if slot.is_saved: 
                    stats[slot.expected_item]["count"] += 1
                else:
                    all_saved = False

        if all_saved: self.has_finished_once = True

        for item in stats:
            if stats[item]["count"] >= stats[item]["total"]:
                stats[item]["count"] = stats[item]["total"]
                stats[item]["done"] = True
        return stats

    def update_camera_state(self):
        """
        Logic cập nhật trạng thái có ưu tiên check vật lạ.
        """
        # 1. Ưu tiên cao nhất: Phát hiện vật sai quy trình (từ tương lai)
        if self.forbidden_item_detected:
            self.cam_state = "false"
            self.status_message = f"WRONG ITEM: {self.forbidden_item_detected}!"
            return

        has_wrong = False
        has_empty = False
        all_ok_now = True

        for s in self.slots.values():
            if s.state == "wrong": has_wrong = True
            if s.state == "empty": has_empty = True
            if s.state != "oke": all_ok_now = False

        if self.has_finished_once:
            self.cam_state = "done"
            self.status_message = "CHECKLIST SAVED"
        elif has_wrong:
            self.cam_state = "false"
            self.status_message = "WRONG ITEM!"
        elif has_empty:
            self.cam_state = "checking"
            self.status_message = "MISSING..."
        elif all_ok_now:
            self.cam_state = "checking"
            self.status_message = "HOLDING 5s..."
        else:
            self.cam_state = "waiting"
            self.status_message = "..."

    def force_reset(self):
        self.has_finished_once = False
        self.cam_state = "waiting"
        self.status_message = "WAITING TRAY"
        self.forbidden_item_detected = None # Reset lỗi vật lạ
        for s in self.slots.values():
            s.reset_state()