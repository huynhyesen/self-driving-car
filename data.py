import csv

input_file = 'driving_log.csv'
output_file = 'driving_log.csv'

cleaned_rows = []
with open(input_file, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        try:
            # Lọc ra các tên file ảnh kết thúc bằng .jpg
            jpgs = [val for val in row if val and val.endswith('.jpg')]
            
            # Lọc ra các giá trị số (steering, throttle, brake, speed) ở cuối dòng
            numeric_vals = []
            for val in row:
                try:
                    numeric_vals.append(float(val))
                except ValueError:
                    continue
            
            if len(jpgs) >= 3 and len(numeric_vals) >= 4:
                # Cắt bỏ đường dẫn rườm rà, chỉ giữ lại định dạng: IMG/tên_file.jpg
                center_img = "IMG/" + jpgs[0].split('/')[-1]
                left_img = "IMG/" + jpgs[1].split('/')[-1]
                right_img = "IMG/" + jpgs[2].split('/')[-1]
                
                steering, throttle, brake, speed = numeric_vals[-4:]
                cleaned_rows.append([center_img, left_img, right_img, steering, throttle, brake, speed])
        except Exception:
            continue

# Ghi ra file CSV mới sạch sẽ
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Ghi thêm tiêu đề cột cho chuyên nghiệp
    writer.writerow(['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    writer.writerows(cleaned_rows)

print(f"Đã làm sạch thành công {len(cleaned_rows)} dòng và lưu vào {output_file}!")