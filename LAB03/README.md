dataset by Aleksandr Sinitca
LAB 1: Regression (ทำนายอายุ)
LinearRegression() อัลกอริทึมพื้นฐานสำหรับปัญหา Regression ใช้เรียนรู้ความสัมพันธ์เชิงเส้นเพื่อทำนายตัวเลขแบบต่อเนื่อง (Continuous Value) เช่น ค่าอายุ
X_pca_simple (1 Feature) vs X_pca_multi (50 Features): Simple Linear Regression: ใช้ Feature เดียวเพื่อทดสอบว่า Feature หลักอันแรกเพียงพอที่จะทำนายอายุได้ไหม Multiple Linear Regression: ใช้ 50 Features เพื่อดูว่าเมื่อเพิ่มข้อมูลมิติภาพมากขึ้น จะช่วยให้ทำนายอายุได้แม่นยำขึ้นหรือไม่
mean_absolute_error (MAE) & r2_score: ใช้ประเมินความคลาดเคลื่อนเฉลี่ยเป็นหน่วยปี (MAE) และวัดว่าโมเดลอธิบายความแปรปรวนของข้อมูลได้ดีแค่ไหน ($R^2$)

LAB 2: Classification (จำแนกเพศ)
LogisticRegression(): อัลกอริทึมมาตรฐานสำหรับแก้ปัญหา Classification เชิงนามธรรม (Binary/Discrete Class) เพื่อจำแนกเพศ (ชาย/หญิง)
class_weight='balanced': ช่วยปรับน้ำหนักข้อมูล ให้โมเดลให้ความสำคัญกับคลาสชายและหญิงเท่าๆ กัน ป้องกันปัญหาโมเดลทายเอนเอียงไปคลาสที่มีจำนวนมากกว่า
accuracy_score, precision, recall, f1_score, confusion_matrix: ตัววัดผลมาตรฐานของ Classification เพื่อดูความแม่นยำ อัตราการทายถูก/ทายผิด และดูตารางการทำนายเปรียบเทียบจริงvsทำนาย
plt.contourf (Decision Boundary): ใช้ PCA 2 มิติย่อข้อมูลลงมาเพื่อวาดกราฟระบายสีแสดง "เส้นแบ่งอาณาเขต" (Decision Boundary) ช่วยให้เห็นภาพชัดเจนว่าโมเดลแบ่งโซนการทำนายชาย/หญิงอย่างไร

LAB 3: Model Comparison (เปรียบเทียบประสิทธิภาพ)
model.score() บน Train vs Test: ใช้เช็กประสิทธิภาพระหว่างชุดข้อมูลที่ใช้ฝึก (Training) กับชุดข้อมูลที่ใช้ทดสอบ (Testing) เพื่อวิเคราะห์ปัญหา Overfitting (ถ้า Train ดีมากแต่ Test แย่) หรือ Underfitting (ถ้าแย่ทั้งคู่)
การเปรียบเทียบ Simple vs Multiple: เพื่อวิเคราะห์ว่าจำนวนตัวแปร (Features) ที่เพิ่มขึ้น ช่วยเพิ่มประสิทธิภาพให้โมเดลเชิงเส้นจริงหรือไม่
