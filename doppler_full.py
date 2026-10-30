import csv
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. CẤU HÌNH
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "dopplerimage"          # Thư mục chứa 20 ảnh đầu vào
OUT = BASE_DIR / "doppler_results"             # Thư mục lưu kết quả
ROI_FILE = OUT / "rois.json"                   # File nhớ vị trí khoanh ROI
SHOW = False                                   # Để False khi chạy 20 ảnh cho nhanh
RESELECT_ROI = True                            # True: cho phép khoanh lại ROI và ghi đè; False: dùng luôn ROI cũ

# Ngưỡng HSV chuẩn (OpenCV: H 0-179, S 0-255, V 0-255)
P = dict(red1=(0, 32), red2=(170, 179), blue=(100, 140),
         s_min=70, v_min=50, kernel=3, min_area=10)

# Khảo sát độ nhạy: mỗi loại 3 mức
SWEEPS = {"s_min": [50, 70, 90],               # Đổi ngưỡng Saturation
          "hue_margin": [-5, 0, 5]}            # Nới/thu hẹp khoảng Hue mỗi bên


# ============================================================
# 2. TÁCH MÀU & ĐẾM VÙNG LIÊN THÔNG (TÁCH RIÊNG MORPH VÀ CC)
# ============================================================
def morph_only(m, p):
    """Chỉ dùng Morphology: Opening (bỏ chấm nhiễu) -> Closing (lấp lỗ nhỏ)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["kernel"],) * 2)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def cc_only(m, p):
    """Chỉ dùng Connected Components: Lọc bỏ các cụm nhỏ (< min_area)."""
    _, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= p["min_area"]
    keep[0] = False                            # Nhãn 0 là nền
    return (keep[lab] * 255).astype(np.uint8)


def count_regions(m):
    """Đếm số vùng màu độc lập trong mask."""
    num_labels, _ = cv2.connectedComponents(m, connectivity=8)
    return max(0, int(num_labels - 1))


def color_masks(hsv, p, stage="full"):
    """
    Trả về mask đỏ và xanh theo 4 chế độ riêng biệt để so sánh:
      - 'raw'  : Trước khi làm sạch (mask thô)
      - 'morph': CHỈ chạy Morphology (Open + Close)
      - 'cc'   : CHỈ chạy Connected Components (lọc diện tích >= min_area trên mask thô)
      - 'full' : Kết hợp cả Morphology + Connected Components
    """
    def in_range(h):
        return cv2.inRange(hsv, np.array([max(0, h[0]), p["s_min"], p["v_min"]], np.uint8),
                                np.array([min(179, h[1]), 255, 255], np.uint8))

    red = in_range(p["red1"]) | in_range(p["red2"])
    blue = in_range(p["blue"])

    if stage == "morph":
        red, blue = morph_only(red, p), morph_only(blue, p)
    elif stage == "cc":
        red, blue = cc_only(red, p), cc_only(blue, p)
    elif stage == "full":
        red, blue = cc_only(morph_only(red, p), p), cc_only(morph_only(blue, p), p)

    blue[red > 0] = 0                          # Pixel trùng thì ưu tiên đỏ
    return red, blue


def calc_dice(m1, m2):
    """Tính độ ổn định mask (hệ số Dice từ 0 đến 1) giữa 2 mask."""
    b1, b2 = m1 > 0, m2 > 0
    s = b1.sum() + b2.sum()
    return round(float(2.0 * np.logical_and(b1, b2).sum() / s), 4) if s > 0 else 1.0


def sweep(hsv, roi_px, base_color, kind, values):
    """Đổi tham số qua 3 mức, trả về (mức, %đỏ, %xanh, %tổng màu, độ ổn định Dice)."""
    rows = []
    for v in values:
        p = dict(P)
        if kind == "s_min":
            p["s_min"] = v
        else:
            p["red1"] = (0, P["red1"][1] + v)
            p["red2"] = (P["red2"][0] - v, 179)
            p["blue"] = (P["blue"][0] - v, P["blue"][1] + v)
        r, b = color_masks(hsv, p, stage="full")
        c = r | b
        pcts = [cv2.countNonZero(m) / roi_px * 100 for m in (r, b, c)]
        rows.append((v, *pcts, calc_dice(base_color, c)))
    return rows


# ============================================================
# 3. VẼ ẢNH & QUẢN LÝ ROI
# ============================================================
def panel(ax, img, title, border="black", cmap=None, vmax=255):
    if cmap is None and img.ndim == 2:
        cmap = "gray"
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(2)


def cutout(roi, mask):
    """Giữ nguyên màu gốc ở vùng mask, nền trắng, vẽ viền xanh lá quanh vùng."""
    out = np.full_like(roi, 255)
    out[mask > 0] = roi[mask > 0]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (0, 200, 0), 1)
    return out


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save(fig, out_folder, name):
    fig.tight_layout()
    fig.savefig(out_folder / name, dpi=120)
    if SHOW:
        plt.show()
    plt.close(fig)


def info_str(m, roi_px):
    """Hiển thị đủ: Diện tích (px) | Tỷ lệ (% ROI) | Số vùng màu."""
    px = cv2.countNonZero(m)
    pct = px / roi_px * 100
    reg = count_regions(m)
    return f"{px:,} px ({pct:.2f}% ROI) | {reg} vùng"


def get_roi(img, name, saved_rois):
    """Mở cửa sổ chọn ROI và ghi đè tọa độ mới (nếu bấm Enter không kéo chuột thì giữ ROI cũ)."""
    if not RESELECT_ROI and name in saved_rois:
        r = saved_rois[name]
        return r["x"], r["y"], r["w"], r["h"]

    print(f"-> Khoanh ROI cho {name} rồi bấm ENTER (hoặc bấm SPACE luôn để giữ ROI cũ nếu đã có)")
    x, y, w, h = cv2.selectROI(f"Chon ROI: {name}", img, showCrosshair=True)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        if name in saved_rois:
            r = saved_rois[name]
            return r["x"], r["y"], r["w"], r["h"]
        return None

    saved_rois[name] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    ROI_FILE.write_text(json.dumps(saved_rois, indent=2, ensure_ascii=False), encoding="utf-8")
    return x, y, w, h


# ============================================================
# 4. XỬ LÝ 1 ẢNH & GHI ĐÈ VÀO THƯ MỤC RIÊNG CỦA ẢNH ĐÓ
# ============================================================
def analyze(path, saved_rois):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    name = path.stem

    # Thư mục con mang tên ảnh (VD: doppler_results/anh1/)
    img_out_dir = OUT / name
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # --- Chọn ROI ---
    roi_coords = get_roi(img, name, saved_rois)
    if roi_coords is None:
        return None
    x, y, w, h = roi_coords
    roi = img[y:y + h, x:x + w]
    roi_px, img_px = w * h, img.shape[0] * img.shape[1]

    # --- BGR -> HSV, tách mask qua 4 chế độ (Raw | Chỉ Morph | Chỉ CC | Kết hợp Full) ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red0, blue0 = color_masks(hsv, P, stage="raw")
    red_m, blue_m = color_masks(hsv, P, stage="morph")
    red_c, blue_c = color_masks(hsv, P, stage="cc")
    red, blue = color_masks(hsv, P, stage="full")

    color_raw = red0 | blue0
    color_morph = red_m | blue_m
    color_cc = red_c | blue_c
    color_full = red | blue
    masks = {"red": red, "blue": blue, "color": color_full, "rest": 255 - color_full}

    # --- Tính diện tích, tỷ lệ % và số vùng màu ---
    row = {"image": name, "roi_px": roi_px, "roi_pct_image": round(roi_px / img_px * 100, 2)}
    for k, m in masks.items():
        px = cv2.countNonZero(m)
        row[f"{k}_px"] = px
        row[f"{k}_pct_roi"] = round(px / roi_px * 100, 2)
        row[f"{k}_pct_image"] = round(px / img_px * 100, 2)

    # Số liệu so sánh riêng biệt: Raw vs Chỉ Morphology vs Chỉ Connected Components vs Kết hợp
    for prefix, c_m, r_m, b_m in [
        ("raw", color_raw, red0, blue0),
        ("morph", color_morph, red_m, blue_m),
        ("cc", color_cc, red_c, blue_c),
        ("final", color_full, red, blue)
    ]:
        row[f"{prefix}_px"] = cv2.countNonZero(c_m)
        row[f"{prefix}_pct_roi"] = round(cv2.countNonZero(c_m) / roi_px * 100, 2)
        row[f"{prefix}_regions"] = count_regions(r_m) + count_regions(b_m)

    row["red_regions"] = count_regions(red)
    row["blue_regions"] = count_regions(blue)

    # --- Ghi đè ảnh Overlay ---
    layer = roi.copy()
    layer[red > 0], layer[blue > 0] = (0, 0, 255), (255, 0, 0)
    overlay = cv2.addWeighted(roi, 0.5, layer, 0.5, 0)
    cv2.imencode(".png", overlay)[1].tofile(str(img_out_dir / f"{name}_overlay.png"))

    # --- Hình 1: ROI + Đỏ + Xanh + Còn lại ---
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    panel(ax[0, 0], rgb(roi), f"1. ROI gốc ({w}x{h}) | {roi_px:,} px | {row['roi_pct_image']}% ảnh gốc")
    for a, k, t, c in [(ax[0, 1], "red", f"2. ĐỎ ({row['red_regions']} vùng)", "red"),
                       (ax[1, 0], "blue", f"3. XANH ({row['blue_regions']} vùng)", "blue"),
                       (ax[1, 1], "rest", "4. CÒN LẠI", "dimgray")]:
        panel(a, rgb(cutout(roi, masks[k])),
              f"{t} | {row[k + '_px']:,} px | {row[k + '_pct_roi']}% ROI", border=c)
    fig.suptitle(f"{name}: Tổng vùng màu = {row['color_px']:,} px ({row['color_pct_roi']}% ROI) "
                 f"| Số vùng màu = {row['final_regions']}")
    save(fig, img_out_dir, f"{name}_1_main.png")

    # --- Hình 2: Kênh R, G, B và H, S, V ---
    b, g, r = cv2.split(roi)
    H, S, V = cv2.split(hsv)
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for a, im, t, cm, vm in [(ax[0, 0], r, "R", "Reds", 255), (ax[0, 1], g, "G", "Greens", 255),
                             (ax[0, 2], b, "B", "Blues", 255), (ax[1, 0], H, "H - Hue", "hsv", 179),
                             (ax[1, 1], S, "S - Saturation", "gray", 255),
                             (ax[1, 2], V, "V - Value", "gray", 255)]:
        panel(a, im, t, cmap=cm, vmax=vm)
    save(fig, img_out_dir, f"{name}_2_rgb_hsv.png")

    # --- Hình 3: Tách riêng Morphology và Connected Components + Overlay (ghi đè file cũ) ---
    fig, ax = plt.subplots(2, 5, figsize=(18, 6.5))
    # Hàng 1: 4 bước của màu Đỏ + Tổng màu hoàn chỉnh
    for a, m, title, col in [
        (ax[0, 0], red0, "1. Đỏ - TRƯỚC lọc", "red"),
        (ax[0, 1], red_m, "2. Đỏ - CHỈ Morphology", "darkorange"),
        (ax[0, 2], red_c, "3. Đỏ - CHỈ Connected Comp.", "teal"),
        (ax[0, 3], red, "4. Đỏ - KẾT HỢP (Morph+CC)", "red"),
        (ax[0, 4], color_full, "5. Tổng màu (Đỏ + Xanh)", "darkgreen")
    ]:
        panel(a, rgb(cutout(roi, m)), f"{title}\n{info_str(m, roi_px)}", border=col)

    # Hàng 2: 4 bước của màu Xanh + Overlay trên ROI gốc
    for a, m, title, col in [
        (ax[1, 0], blue0, "1. Xanh - TRƯỚC lọc", "blue"),
        (ax[1, 1], blue_m, "2. Xanh - CHỈ Morphology", "darkorange"),
        (ax[1, 2], blue_c, "3. Xanh - CHỈ Connected Comp.", "teal"),
        (ax[1, 3], blue, "4. Xanh - KẾT HỢP (Morph+CC)", "blue")
    ]:
        panel(a, rgb(cutout(roi, m)), f"{title}\n{info_str(m, roi_px)}", border=col)

    panel(ax[1, 4], rgb(overlay),
          f"Overlay hoàn chỉnh\n{row['final_px']:,} px ({row['final_pct_roi']}% ROI) | {row['final_regions']} vùng",
          border="purple")

    fig.suptitle(
        f"{name} - So sánh tổng màu: Trước lọc ({row['raw_px']:,} px, {row['raw_pct_roi']}%, {row['raw_regions']} vùng) | "
        f"Chỉ Morphology ({row['morph_px']:,} px, {row['morph_pct_roi']}%, {row['morph_regions']} vùng) | "
        f"Chỉ CC ({row['cc_px']:,} px, {row['cc_pct_roi']}%, {row['cc_regions']} vùng) | "
        f"Kết hợp Morph+CC ({row['final_px']:,} px, {row['final_pct_roi']}%, {row['final_regions']} vùng)",
        fontsize=9.5
    )
    save(fig, img_out_dir, f"{name}_3_morphology_overlay.png")

    # --- Hình 4: Độ nhạy theo ngưỡng ---
    sens = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    line_configs = [("Đỏ", "#d62728"), ("Xanh", "#1f77b4"), ("Tổng màu", "#2ca02c")]

    for ax_, (kind, vals) in zip(axs, SWEEPS.items()):
        rows = sweep(hsv, roi_px, color_full, kind, vals)
        sens += [(name, kind, *r_) for r_ in rows]
        for i, (lab, color_code) in enumerate(line_configs):
            y_vals = [r_[i + 1] for r_ in rows]
            ax_.plot(vals, y_vals, marker="o", linewidth=2, color=color_code, label=lab)
        ax_.set(xlabel=kind, ylabel="% ROI", title=f"Độ nhạy theo {kind}")
        ax_.set_xticks(vals)
        ax_.grid(alpha=0.3)
        ax_.legend()
        ax_.set_ylim(bottom=0, top=max(40, ax_.get_ylim()[1] * 1.15))

    save(fig, img_out_dir, f"{name}_4_sensitivity.png")
    return row, sens


# ============================================================
# 5. MAIN: CHẠY HÀNG LOẠT 20 ẢNH & GHI ĐÈ BẢNG TỔNG HỢP
# ============================================================
def save_summary_stats(rows, path):
    """Tự động tính Trung bình ± Độ lệch chuẩn (Mean ± SD) và Trung vị (IQR) cho 20 ảnh."""
    if not rows:
        return
    metrics = [
        "red_pct_roi", "blue_pct_roi", "color_pct_roi",
        "raw_pct_roi", "morph_pct_roi", "cc_pct_roi", "final_pct_roi",
        "raw_regions", "morph_regions", "cc_regions", "final_regions"
    ]
    stat_rows = []
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        mean_v, sd_v = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        med_v = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        stat_rows.append({
            "chi_so": m,
            "so_anh_n": len(vals),
            "trung_binh_ ± _do_lech_chuan": f"{mean_v:.2f} ± {sd_v:.2f}",
            "trung_vi_(IQR)": f"{med_v:.2f} ({q25:.2f} - {q75:.2f})",
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2)
        })
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys()))
        wr.writeheader()
        wr.writerows(stat_rows)


def main():
    IMAGE_DIR.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    # Ghi đè thông số ngưỡng
    (OUT / "thresholds.json").write_text(
        json.dumps({"hsv_params": P, "sensitivity": SWEEPS}, indent=2), encoding="utf-8")

    # Đọc tọa độ ROI đã lưu (nếu có)
    saved_rois = json.loads(ROI_FILE.read_text(encoding="utf-8")) if ROI_FILE.exists() else {}

    # Quét tất cả ảnh trong thư mục dopplerimage theo đúng thứ tự anh1 -> anh20
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    image_list = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in exts],
        key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', p.stem)]
    )

    if not image_list:
        print(f"[THÔNG BÁO] Bạn hãy bỏ 20 ảnh vào thư mục: {IMAGE_DIR.resolve()} rồi chạy lại nhé!")
        return

    all_rows, all_sens = [], []
    print(f"=== ĐANG XỬ LÝ {len(image_list)} ẢNH TRONG '{IMAGE_DIR.name}' ===")

    for idx, img_path in enumerate(image_list, 1):
        res = analyze(img_path, saved_rois)
        if res is None:
            print(f"[{idx}/{len(image_list)}] Bỏ qua: {img_path.name}")
            continue
        row, sens = res
        all_rows.append(row)
        all_sens.extend(sens)
        print(f"[{idx:02d}/{len(image_list):02d}] Đã ghi đè: {row['image']:<8} "
              f"| Raw: {row['raw_regions']:>2} vùng | Chỉ Morph: {row['morph_regions']:>2} vùng "
              f"| Chỉ CC: {row['cc_regions']:>2} vùng | Morph+CC: {row['final_regions']:>2} vùng")

    if not all_rows:
        return

    # 1. Ghi đè bảng kết quả của toàn bộ 20 ảnh
    with open(OUT / "ket_qua_doppler.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        wr.writeheader()
        wr.writerows(all_rows)

    # 2. Ghi đè bảng độ nhạy (kèm chỉ số ổn định Dice)
    fields = ["image", "param", "value", "red_pct_roi", "blue_pct_roi", "color_pct_roi", "dice_stability"]
    sens_rows = [dict(zip(fields, (s[0], s[1], s[2], round(s[3], 2), round(s[4], 2), round(s[5], 2), s[6])))
                 for s in all_sens]
    with open(OUT / "sensitivity.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(sens_rows)

    # 3. Ghi đè bảng thống kê Trung bình ± Độ lệch chuẩn của 20 ảnh
    save_summary_stats(all_rows, OUT / "thong_ke_trung_binh.csv")

    print(f"\n=== HOÀN TẤT! Đã ghi đè toàn bộ kết quả mới vào: {OUT.resolve()} ===")


if __name__ == "__main__":
    main()