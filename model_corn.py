from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

corn = Blueprint('corn', __name__)

# تحميل نموذج الذرة
interpreter = tf.lite.Interpreter(model_path=r'D:\\Proggraming\\AI\\plantsage\\maize_model.tflite')
interpreter.allocate_tensors()

# أسماء الفئات باللغة العربية
class_names = {
    0: "بقعة ورقية (السديم الرمادي)",
    1: "صدأ شائع",
    2: "عفن الأوراق الشمالية",
    3: "صحة جيدة"
}

# دالة لتحويل الصورة
def prepare_image(image, target_size):
    image = image.convert("RGB")  # تحويل الصورة إلى RGB
    if image.size != target_size:
        image = image.resize(target_size)  # تغيير الحجم إذا كان مختلفًا
    image = np.array(image) / 255.0  # تحويل القيم إلى النطاق [0, 1]
    image = np.expand_dims(image, axis=0)  # إضافة بُعد إضافي
    return image.astype(np.float32)  # تحويل النوع إلى FLOAT32

# دالة لإرجاع الأعراض والعلاج
def get_symptoms_and_treatment(predicted_class_index):
    symptoms_treatment = {
        0: {
            'symptoms': (
                "1 - أعراض الإصابة: ظهور بقع رمادية على الأوراق مع انتشارها.\n"
                "2 - المسبب المرضي: Alternaria solani.\n"
                "3 - دورة حياة الفطر: ينمو الفطر في ظروف رطبة، حيث ينتشر عن طريق البذور والتربة.\n"
                "4 - المقاومة و المكافحة: استخدم مبيدات فطرية مثل كابتان ومانكوزب. "
                "تجنب الرطوبة الزائدة وراقب المحاصيل."
            ),
            'treatment': "مبيدات فطرية مثل:\n1. كابتان\n2. مانكوزب."
        },
        1: {
            'symptoms': (
                "1 - أعراض الإصابة: ظهور بقع حمراء أو برتقالية على الأوراق.\n"
                "2 - المسبب المرضي: Puccinia sorghi.\n"
                "3 - دورة حياة الفطر: ينتشر عن طريق الأبواغ في الهواء.\n"
                "4 - المقاومة و المكافحة: استخدم مبيدات مثل مركب النحاس وكلوريد الصوديوم. "
                "قم بإزالة الأوراق المصابة."
            ),
            'treatment': "مبيدات مثل:\n1. مركب النحاس\n2. كلوريد الصوديوم."
        },
        2: {
            'symptoms': (
                "1 - أعراض الإصابة: ظهور عفن أسود على الأوراق مع ذبول النبات.\n"
                "2 - المسبب المرضي: Fusarium spp.\n"
                "3 - دورة حياة الفطر: يعيش في التربة ويمتد عن طريق الجذور.\n"
                "4 - المقاومة و المكافحة: استخدم مبيدات فطرية مثل توبرا وفيندازول. "
                "تجنب الري من الأعلى."
            ),
            'treatment': "مبيدات فطرية مثل:\n1. توبرا\n2. فيندازول."
        },
        3: {
            'symptoms': (
                "1 - أعراض الإصابة: النبات في حالة جيدة.\n"
                "2 - المسبب المرضي: لا يوجد.\n"
                "3 - دورة حياة الفطر: لا يوجد.\n"
                "4 - المقاومة و المكافحة: استمر في العناية به."
            ),
            'treatment': "استمر في العناية به."
        }
    }
    return symptoms_treatment.get(predicted_class_index, {'symptoms': 'غير معروف', 'treatment': 'غير معروف'})

# نقطة النهاية للتنبؤ
@corn.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'مفيش صورة تم تقديمها'}), 400

    file = request.files['file']

    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'error': 'فشل في فتح الصورة: ' + str(e)}), 400

    # إعداد الصورة للتنبؤ
    try:
        prepared_image = prepare_image(image, target_size=(256, 256))
    except Exception as e:
        return jsonify({'error': 'فشل في إعداد الصورة: ' + str(e)}), 400

    # إجراء التنبؤ
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], prepared_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        return jsonify({'error': 'فشل في إجراء التنبؤ: ' + str(e)}), 500

    # استخراج المخرجات المطلوبة
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions) * 100
    predicted_class_name = class_names.get(predicted_class_index, "غير معروف")
    symptoms_and_treatment = get_symptoms_and_treatment(predicted_class_index)

    # إعداد المخرجات
    output = {
        'predicted_class_index': int(predicted_class_index),
        'predicted_class_name': predicted_class_name,
        'confidence_score': f"{confidence_score:.2f}%",
        'symptoms': symptoms_and_treatment['symptoms'],
        'treatment': symptoms_and_treatment['treatment']
    }

    return jsonify(output)
