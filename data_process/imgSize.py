import os
from PIL import Image

ROOT_DIR = r"D:\ui2code\InteractionSequence\data_process\mobileViewData\com.nordicusability.jiffy\states"
DST_ROOT = r"D:\ui2code\InteractionSequence\data_process\mobileViewData\nordicusability\jiffy"
CROP_HEIGHT = 140
CROP_TOP = 70
CROP_BOTTOM = 140
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

def crop_and_save(src_path, dst_path):
    img = Image.open(src_path)
    w, h = img.size



    if h <= CROP_BOTTOM:
        print(f"⚠️ 跳过（高度不足）: {src_path} ({w}x{h})")
        return

    # ✅ 强制输出为 PNG
    dst_path = os.path.splitext(dst_path)[0] + ".png"

    # cropped = img.crop((0, 0, w, h - CROP_HEIGHT))
    # cropped = img.crop((0, CROP_TOP, w, h))
    cropped = img.crop((0, CROP_TOP, w, h - CROP_BOTTOM))
    # ✅ 只裁剪底部
    # cropped = img.crop((0, 0, w, h - CROP_BOTTOM))

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cropped.save(dst_path)

    print(f"✅ {src_path} → {dst_path} ({w}x{h} → {w}x{h-CROP_BOTTOM})")

def process_all_images(src_root, dst_root):
    for root, dirs, files in os.walk(src_root):
        if os.path.basename(root) != "states":
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in IMAGE_EXTS:
                continue

            src_path = os.path.join(root, file)
            # rel_path = os.path.relpath(src_path, src_root)
            dst_path = os.path.join(dst_root, file)
            # parts = rel_path.split(os.sep)

            
            crop_and_save(src_path, dst_path)

            # # 去掉 states
            # parts = [p for p in parts if p != "states"]
            # rel_path_no_states = os.path.join(*parts)
            # dst_path = os.path.join(dst_root, rel_path_no_states)

            # crop_and_save(src_path, dst_path)

if __name__ == "__main__":
    process_all_images(ROOT_DIR, DST_ROOT)
    print("\n🎉 所有图片已处理并保存到")
