from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

script_dir = Path(__file__).parent.resolve()

model_file = script_dir/'models'/'cloud-nocloud.tflite' # name of model
label_file = script_dir/'models'/'labels.txt' # Name of your label file
image_file = script_dir/'..'/'Image'/'picture'/'coscodingofspace_photo_0653.jpg' # Name of image for classification

interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

size = common.input_size(interpreter) 
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

labels = read_label_file(label_file)
for c in classes:
    print(f'{labels.get(c.id, c.id)} {c.score:.5f}')

    