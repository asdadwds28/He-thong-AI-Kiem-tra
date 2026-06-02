import numpy as np
from slot_recovery import SlotRecovery
from utils import GeometryUtils

class FrameProcessor:
    def __init__(self, cam_config, conf_threshold=0.5):
        self.cam_config = cam_config
        self.conf_threshold = conf_threshold
        self.recovery = SlotRecovery()
        self.geo_utils = GeometryUtils()

    def process(self, results_slot, results_item):
        """
        Xử lý logic cho 1 camera.
        Thêm logic check vật sai quy trình (Forbidden Item Check).
        """
        # --- BƯỚC 1: XỬ LÝ SLOT (Tìm khay) ---
        slots_found = []
        slot_centers = []
        
        if hasattr(results_slot, 'obb') and results_slot.obb is not None:
            obbs = results_slot.obb.xyxyxyxy.cpu().numpy()
            confs = results_slot.obb.conf.cpu().numpy()
            for obb, conf in zip(obbs, confs):
                if conf < self.conf_threshold: continue
                center = np.mean(obb, axis=0)
                slot_centers.append(center)
                slots_found.append(obb)

        is_tray_detected = len(slot_centers) >= 3

        # --- BƯỚC 2: ĐỊNH DANH & CẬP NHẬT VỊ TRÍ ---
        geometry_ids = self.geo_utils.identify_slots_logic(slot_centers)
        if geometry_ids:
            if len(geometry_ids) == 5:
                self.recovery.update_reference(geometry_ids)
            elif len(geometry_ids) >= 2:
                geometry_ids = self.recovery.recover(geometry_ids)

            for local_id, center_pos in geometry_ids.items():
                slot_obj = self.cam_config.get_slot_by_local_id(local_id)
                if slot_obj:
                    if slots_found:
                        closest_obb = min(slots_found, key=lambda obb: np.linalg.norm(np.mean(obb, axis=0) - center_pos))
                        slot_obj.update_position(closest_obb, center_pos)

        # --- BƯỚC 3: KIỂM TRA ITEM ---
        items_boxes = []
        items_classes = []
        
        # Reset cờ báo vật lạ trước khi check
        self.cam_config.forbidden_item_detected = None

        if hasattr(results_item, 'boxes') and results_item.boxes:
            boxes = results_item.boxes.xyxy.cpu().numpy()
            confs = results_item.boxes.conf.cpu().numpy()
            clss = results_item.boxes.cls.cpu().numpy()
            
            detected_classes_set = set() # Dùng set để check nhanh

            for box, conf, cls in zip(boxes, confs, clss):
                if conf < 0.45: continue 
                
                cls_name = results_item.names[int(cls)]
                items_boxes.append(box)
                items_classes.append(cls_name)
                detected_classes_set.add(cls_name)

            # --- [LOGIC MỚI] CHECK VẬT SAI QUY TRÌNH ---
            # Chỉ check nếu có detect được khay (tránh báo lỗi khi chưa có gì)
            if is_tray_detected:
                # Tìm xem có vật nào detect được mà KHÔNG nằm trong allowed list không?
                # allowed_classes đã được tính tích lũy bên config.py
                forbidden_items = detected_classes_set - self.cam_config.allowed_classes
                
                if forbidden_items:
                    # Lấy tên vật lạ đầu tiên để báo lỗi
                    wrong_item = list(forbidden_items)[0]
                    self.cam_config.forbidden_item_detected = wrong_item
                    
                    # Nếu phát hiện vật lạ -> Cập nhật trạng thái ngay lập tức và return
                    # (Hoặc vẫn để chạy tiếp để vẽ box nhưng state sẽ bị đè là False)
                    # Ở đây ta cho chạy tiếp để visualizer vẫn vẽ được box đỏ của slot
        
        # --- BƯỚC 4: CHECK VA CHẠM SLOT ---
        for s_id, slot in self.cam_config.slots.items():
            if slot.obb_points is None: continue 

            is_occupied = False
            for box, cls_name in zip(items_boxes, items_classes):
                if self.geo_utils.is_item_in_slot(box, slot.obb_points, threshold=0.45):
                    is_occupied = True
                    if cls_name == slot.expected_item:
                        slot.set_state("oke", cls_name)
                    else:
                        slot.set_state("wrong", cls_name)
                    break 
            
            if not is_occupied:
                slot.set_state("empty")

        # --- BƯỚC 5: UPDATE TRẠNG THÁI CAMERA ---
        # Hàm này sẽ ưu tiên check forbidden_item_detected trước
        self.cam_config.update_camera_state()
        
        return is_tray_detected