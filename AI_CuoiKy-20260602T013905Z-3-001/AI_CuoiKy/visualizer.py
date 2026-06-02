import cv2
import numpy as np
import time

class Visualizer:
    def __init__(self):
        self.prev_time = 0
        # Bảng màu (BGR)
        self.colors = {
            "empty": (180, 180, 180), # Xám nhạt
            "oke": (0, 255, 0),       # Xanh lá tươi
            "locked": (0, 200, 0),    # Xanh lá đậm (Đã Saved)
            "wrong": (0, 0, 255),     # Đỏ
            "checking": (0, 255, 255),# Vàng
            "text": (255, 255, 255),  
            "debug": (255, 0, 255),   
            "dim_text": (100, 100, 100), 
            "highlight": (0, 255, 255),  
            "bg": (0, 0, 0)
        }

    def draw_fps(self, frame):
        """Vẽ FPS lên góc trái trên"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        
        cv2.rectangle(frame, (10, 10), (150, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def draw_slot_obb(self, frame, slot):
        if slot.obb_points is None: return

        # Mặc định theo state hiện tại (thực tế)
        color = self.colors.get(slot.state, (255, 255, 255))
        label = f"S{slot.id}"
        thickness = 2

        # 1. NẾU ĐANG CÓ VẬT (OKE)
        if slot.state == "oke":
            if slot.is_saved:
                # Đã Save và vật vẫn đang ở đó -> Xanh đậm, hiện SAVED
                color = self.colors["locked"] 
                label += " SAVED"
                thickness = 3
            elif slot.first_oke_time is not None:
                # Đang đếm ngược
                elapsed = time.time() - slot.first_oke_time
                remaining = max(0.0, 3.0 - elapsed)
                label += f" {remaining:.1f}s"
                if int(elapsed * 10) % 2 == 0: color = (150, 255, 150)
            else:
                label += " OK"

        # 2. NẾU TRỐNG (EMPTY)
        elif slot.state == "empty":
            # Dù đã save hay chưa, nếu trống thì báo trống (để công nhân biết mà bỏ lại)
            color = self.colors["empty"]
            # Không hiện chữ SAVED ở đây để tránh hiểu nhầm
            # Nếu muốn nhắc nhở đã save, có thể thêm dấu (v) nhỏ, nhưng yêu cầu của bạn là "xóa không lưu lại"
            pass 

        # 3. NẾU SAI
        elif slot.state == "wrong":
            label += " X"

        # Vẽ
        pts = slot.obb_points.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, thickness)
        
        c = slot.center
        text_pos = (c[0] - 40, c[1] + 5)
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_item_box(self, frame, box, label, conf):
        """Vẽ box vật thể item detect được"""
        x1, y1, x2, y2 = map(int, box)
        color = (255, 165, 0) # Màu cam
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        caption = f"{label} {int(conf*100)}%"
        (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, caption, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_camera_info(self, frame, cam_config):
        """Vẽ thông tin cam ở góc dưới"""
        h, w = frame.shape[:2]
        # Nền đen đơn giản (nhanh hơn addWeighted)
        cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
        
        st_color = self.colors.get(cam_config.cam_state, (255, 255, 255))
        if cam_config.cam_state == "done": st_color = (0, 255, 0)
        if cam_config.cam_state == "false": st_color = (0, 0, 255)

        text_info = f"{cam_config.cam_name.upper()} | {cam_config.cam_state.upper()}"
        cv2.putText(frame, text_info, (10, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, st_color, 2)
        
        cv2.putText(frame, cam_config.status_message, (10, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def draw_dashboard_on_roi(self, roi, all_configs):
        """
        Vẽ checklist trực tiếp lên vùng ảnh Dashboard (bên phải)
        """
        # roi là ảnh (H, W, 3) nền đen
        x_start = 20
        y_cursor = 50
        
        cv2.putText(roi, "CHECKLIST STATUS", (x_start, y_cursor), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(roi, (x_start, y_cursor + 10), (roi.shape[1] - 20, y_cursor + 10), (255, 255, 255), 1)
        y_cursor += 40

        for cfg in all_configs:
            is_active = cfg.cam_state != "waiting"
            
            if is_active:
                header_color = self.colors["highlight"]
                thickness = 2
            else:
                header_color = self.colors["dim_text"]
                thickness = 1
            
            # Tên Cam
            cam_label = f"> {cfg.cam_name.upper()}"
            cv2.putText(roi, cam_label, (x_start, y_cursor), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, thickness)
            y_cursor += 25
            
            # Item Stats
            stats = cfg.get_item_counts()
            for item_name, info in stats.items():
                # Format: "  Den_nho: 2/3"
                count_str = f"  {item_name}: {info['count']}/{info['total']}"
                
                if info['done']:
                    item_color = (0, 255, 0)
                    count_str += " OK"
                elif is_active:
                    item_color = (255, 255, 255)
                    # Nếu cam đang False -> Chữ đỏ
                    if cfg.cam_state == "false": item_color = (0, 0, 255)
                else:
                    item_color = self.colors["dim_text"]

                cv2.putText(roi, count_str, (x_start, y_cursor), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, item_color, 1)
                y_cursor += 20
            
            y_cursor += 15