from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

sugar_cane = Blueprint('sugar_cane', __name__)

# تحميل نموذج القصب
model = keras.models.load_model(r'D:\\Proggraming\\AI\\SugarCane_HealthyDetective\\keras_model.h5')  # نموذج القصب

# أسماء الفئات باللغة العربية
class_names = {
    0: "صحة جيدة",
    1: "مرض البياض الدقيقي",
    2: "مرض العفن الأسود",
    3: "مرض الفيوزاريوم",
    4: "مرض الصدأ",
    5: "مرض ورق القصب الأصفر"
}

# دالة لتحويل الصورة
def prepare_image(image, target_size):
    image = image.convert("RGB")  # تحويل الصورة إلى RGB
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# نقطة النهاية للتنبؤ
@sugar_cane.route('/predict', methods=['POST'])
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
        prepared_image = prepare_image(image, target_size=(224, 224))
    except Exception as e:
        return jsonify({'error': 'فشل في إعداد الصورة: ' + str(e)}), 400

    # إجراء التنبؤ
    try:
        predictions = model.predict(prepared_image)
    except Exception as e:
        return jsonify({'error': 'فشل في إجراء التنبؤ: ' + str(e)}), 500

    # استخراج المخرجات المطلوبة
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions) * 100
    predicted_class_name = class_names.get(predicted_class_index, "غير معروف")

    # استخراج الأعراض والعلاج والمقاومة والمسبب المرضي من مخرجات النموذج
    if predicted_class_index == 1:  # مرض البياض الدقيقي
        symptoms = "مرض البياض الدقيقي يظهر على شكل بقع بيضاء أو زغابات قطنية على السطح العلوي للأوراق. قد يؤدي إلى ضعف النمو وذبول النباتات."
        treatment = ("لعلاج هذا المرض، يُوصى باستخدام مبيدات فطرية فعالة مثل:\n"
                     "1. فندازول (Fungizone): يستخدم لمكافحة العديد من الأمراض الفطرية.\n"
                     "2. كابتان (Captan): يستخدم لحماية النباتات من الفطريات.\n"
                     "3. ميتالاكسي (Metalaxyl): فعال ضد الفطريات المائية.")
        resistance = "يمكن استخدام أصناف مقاومة معروفة في الزراعة أو زراعة أصناف جديدة تتحمل هذا المرض."
        pathogen = "المسبب المرضي: الفطر Erysiphe spp. الذي يزدهر في ظروف الرطوبة العالية."

    elif predicted_class_index == 2:  # مرض العفن الأسود
        symptoms = "يظهر مرض العفن الأسود على شكل بقع سوداء على الأوراق، مما يؤدي إلى تدهور النبات. قد تظهر الأعراض أيضًا على السيقان."
        treatment = ("لعلاج هذا المرض، يجب إزالة الأجزاء المريضة من النبات ورش المبيدات مثل:\n"
                     "1. توبرا (Topsin): يساعد في مكافحة الفطريات.\n"
                     "2. سيفين (Sevin): مبيد حشري ومبيد فطري فعال.")
        resistance = "اختيار أصناف مقاومة وزراعة نباتات صحية يمكن أن تساعد في تقليل الإصابة."
        pathogen = "المسبب المرضي: الفطر Ascochyta spp. الذي يفضل الرطوبة العالية."

    elif predicted_class_index == 3:  # مرض الفيوزاريوم
        symptoms = "يتميز مرض الفيوزاريوم بذبول النباتات وسقوط الأوراق، مما يؤدي إلى ضعف النبات."
        treatment = ("من المهم تحسين صرف المياه حول النباتات وتطبيق مبيدات فطرية مثل:\n"
                     "1. فندازول (Fungizone): يقضي على الفطريات الضارة.")
        resistance = "استخدام أصناف مقاومة وتطبيق ممارسات زراعية جيدة مثل تحسين الصرف."
        pathogen = "المسبب المرضي: الفطر Fusarium spp. الذي يسبب تدهور صحة النبات في ظروف مائية."

    elif predicted_class_index == 4:  # مرض الصدأ
        symptoms = "يتسبب مرض الصدأ في ظهور بقع صدئية على الأوراق، مما يقلل من قدرة النبات على التمثيل الضوئي."
        treatment = ("يمكن استخدام المبيدات الخاصة بمرض الصدأ مثل:\n"
                     "1. أفلوكس (Afloc): فعال ضد الفطريات.\n"
                     "2. مرشات خاصة بالصدأ: تساعد في السيطرة على المرض.")
        resistance = "اختيار أصناف مقاومة وزراعة نباتات صحية هو الحل الأمثل لمكافحة هذا المرض."
        pathogen = "المسبب المرضي: الفطر Puccinia spp. الذي ينتشر في ظروف رطبة."

    elif predicted_class_index == 5:  # مرض ورق القصب الأصفر
        symptoms = "يظهر هذا المرض على شكل اصفرار عام للأوراق، مما يؤثر على إنتاجية النبات."
        treatment = ("تعديل الممارسات الزراعية ضروري، واستخدام مركبات النيتروجين لتحسين صحة النباتات مثل:\n"
                     "1. تحسين صرف المياه لمنع تجمع الماء حول الجذور.")
        resistance = "تجنب زراعة الأصناف الحساسة والتأكد من صحة الشتلات قبل الزراعة."
        pathogen = "المسبب المرضي: فيروس يسبب تدهورًا عامًا في صحة النبات."

    else:  # صحة جيدة
        symptoms = "النبات في حالة جيدة، ولا تظهر عليه أي أعراض مرضية."
        treatment = "استمر في العناية به وزراعته بشكل صحيح، مع توفير جميع متطلبات النمو."
        resistance = "لا حاجة لأي إجراءات خاصة، فقط استمر في المتابعة."
        pathogen = "لا يوجد."

    # إعداد المخرجات
    output = {
        'predicted_class_index': int(predicted_class_index),
        'predicted_class_name': predicted_class_name,
        'confidence_score': f"{confidence_score:.2f}%",
        'symptoms': symptoms,
        'treatment': treatment,
        'resistance': resistance,
        'pathogen': pathogen
    }

    return jsonify(output)
