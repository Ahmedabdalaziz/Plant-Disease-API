from flask import Flask, jsonify, request
from model_sugar_cane import sugar_cane
from model_tomato import tomato
from model_potato import potato
from model_corn import corn  # استيراد نموذج الذرة

app = Flask(__name__)

# تسجيل Blueprints الخاصة بالنماذج
app.register_blueprint(sugar_cane, url_prefix='/sugar_cane')
app.register_blueprint(tomato, url_prefix='/tomato')
app.register_blueprint(potato, url_prefix='/potato')
app.register_blueprint(corn, url_prefix='/corn')  # تسجيل نموذج الذرة

# نقطة النهاية لاختيار النبات
@app.route('/predict', methods=['POST'])
def predict():
    plant_type = request.form.get('plant_type')  # الحصول على نوع النبات
    if not plant_type:
        return jsonify({'error': 'يرجى تقديم نوع النبات مثل: sugar_cane, tomato, potato, corn.'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'مفيش صورة تم تقديمها. يرجى رفع صورة للنبات.'}), 400

    file = request.files['file']

    # توجيه الصورة إلى النموذج المناسب بناءً على نوع النبات
    if plant_type == 'sugar_cane':
        response = sugar_cane.predict(file)
    elif plant_type == 'tomato':
        response = tomato.predict(file)
    elif plant_type == 'potato':
        response = potato.predict(file)
    elif plant_type == 'corn':
        response = corn.predict(file)  # توجيه الصورة إلى نموذج الذرة
    else:
        return jsonify({'error': 'نوع النبات غير مدعوم. يرجى اختيار واحد من الأنواع المدعومة.'}), 400

    return response

# التعامل مع الأخطاء
@app.errorhandler(401)
def unauthorized_error(error):
    return jsonify({'error': 'وصول غير مصرح به: ' + str(error)}), 401

@app.errorhandler(Exception)
def handle_exception(error):
    return jsonify({'error': 'حدث خطأ: ' + str(error)}), 500

# تشغيل السيرفر
if __name__ == '__main__':
    app.run(debug=True)
