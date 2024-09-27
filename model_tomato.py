from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

tomato = Blueprint('tomato', __name__)

# تحميل نموذج الطماطم
interpreter = tf.lite.Interpreter(model_path=r'D:\\Proggraming\\AI\\plantsage\\tomato_model.tflite')
interpreter.allocate_tensors()

# أسماء الفئات باللغة العربية
class_names = {
    0: "بقعة بكتيرية",
    1: "عفن مبكر",
    2: "عفن متأخر",
    3: "قالب الورقة",
    4: "بقعة ورقية سبطورية",
    5: "عنكبوتية",
    6: "بقعة مستهدفة",
    7: "فيروس تجاعيد أوراق الطماطم",
    8: "فيروس الفسيفساء",
    9: "صحة جيدة"
}

# دالة لتحويل الصورة
def prepare_image(image, target_size):
    image = image.convert("RGB")  # تحويل الصورة إلى RGB
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # تحويل القيم إلى النطاق [0, 1]
    image = image.astype(np.float32)  # تأكد من أن النوع هو FLOAT32
    image = np.expand_dims(image, axis=0)  # إضافة بُعد إضافي
    return image

# دالة لإرجاع الأعراض والعلاج
def get_symptoms_and_treatment(predicted_class_index):
    symptoms_treatment = {
        0: {
            'symptoms': "ظهور بقع مائية صغيرة على الأوراق، مما يؤدي إلى ذبول الأوراق.",
            'treatment': "استخدم مبيدات فطرية مثل:\n1. كوبر \n2. فطريات المانكو جوب.\n" \
                        "تجنب الري من الأعلى وقلل من الرطوبة في البيئات المحيطة."
        },
        1: {
            'symptoms': "بقع سوداء أو بنية على الأوراق السفلية مع اصفرار وذبول للأوراق.",
            'treatment': "استخدم مبيدات فطرية مثل:\n1. كابتان \n2. توبرا.\n" \
                        "حافظ على تباعد النباتات لضمان تدفق الهواء."
        },
        2: {
            'symptoms': "ظهور بقع داكنة على الأوراق مع وجود عفن على الفاكهة.",
            'treatment': "استخدم مبيدات فطرية مثل:\n1. مانكوزب \n2. توبرا.\n" \
                        "تجنب الرطوبة العالية والري في المساء."
        },
        3: {
            'symptoms': "تظهر أوراق عفنة مع وجود رائحة عفنة.",
            'treatment': "أزل الأوراق المريضة فوراً واستخدم:\n1. بوردو ميكس \n2. مبيدات فطرية."
        },
        4: {
            'symptoms': "بقع دائرية بنية أو رمادية على الأوراق مع تجاعيد.",
            'treatment': "استخدم مبيدات مثل:\n1. مانكوزب \n2. ديفين.\n" \
                        "قم بتحسين الصرف والتهوية."
        },
        5: {
            'symptoms': "تظهر نقاط بيضاء صغيرة على الأوراق وتبدو الأوراق باهتة.",
            'treatment': "استخدم مبيدات حشرية مثل:\n1. كاراتيه \n2. سبينوساد.\n" \
                        "قم برش النبات بشكل دوري."
        },
        6: {
            'symptoms': "تظهر بقع دائرية بنية مع وجود حلقة صفراء.",
            'treatment': "استخدم مبيدات فطرية مثل:\n1. فيندازول \n2. توبرا."
        },
        7: {
            'symptoms': "تجعيد الأوراق واصفرارها مع ضعف في النمو.",
            'treatment': "لا يوجد علاج محدد، قم بإزالة النباتات المصابة لمنع انتشار الفيروس."
        },
        8: {
            'symptoms': "ظهور بقع متفرقة وألوان مختلفة على الأوراق.",
            'treatment': "قم بإزالة النباتات المصابة واتباع إجراءات العناية."
        },
        9: {
            'symptoms': "النبات في حالة جيدة.",
            'treatment': "استمر في العناية به."
        }
    }
    return symptoms_treatment.get(predicted_class_index, {'symptoms': 'غير معروف', 'treatment': 'غير معروف'})

# نقطة النهاية للتنبؤ
@tomato.route('/predict', methods=['POST'])
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
