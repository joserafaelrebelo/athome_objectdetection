from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx

# set context
ctx = mx.gpu()

# load model
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)

# load input image
im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                        'mxnet-ssd/master/data/demo/dog.jpg',
                        path='dog.jpg')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
x = x.as_in_context(ctx)

# call forward and show plot
class_IDs, scores, bounding_boxs = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                        class_IDs[0], class_names=net.classes)
plt.show()