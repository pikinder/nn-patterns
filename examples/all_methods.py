import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

import nn_patterns
import nn_patterns.utils.tests.networks.imagenet
import lasagne.nonlinearities
import lasagne
import theano

import imp
base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))

import pickle
def load(filename):
    f = np.load(filename)
    ret = [f["arr_%i" % i] for i in range(len(f.keys()))]
    return ret

def loadp(filename):
    f = np.load(filename)
    ret = {"A": [f["A_%i" % i] for i in range(16)]}
    return ret

if __name__ == "__main__":
    images, label_to_class_name = eutils.get_imagenet_data()

    target = "max_output"
    methods = {
        # NAME         POSTPROCESSING          TITLE              (GROUP)INDEX

        # Show input.
        "input":      (eutils.original_image,  ("", "Input"),     (0, 0)      ),

        # Function
        "gradient":   (eutils.back_projection, ("", "Gradient"),  (1, 1)      ),

        # Signal
        "deconvnet":  (eutils.back_projection, ("", "DeConvNet"), (2, 2)      ),
        "guided":     (eutils.back_projection, ("Guided",
                                                  "Backprop"),    (2, 3)      ),
        "patternnet": (eutils.back_projection, ("PatterNet",
                                                  "($S_{a+-}$)"), (2, 4)      ),

        # Interaction
        "patternlrp": (eutils.heatmap,         ("PatternLRP",
                                                  "($S_{a+-}$)"), (3, 5)      ),
        "lrp.z":      (eutils.heatmap,         ("", "LRP"),       (3, 6)      ),
    }

    ###########################################################################
    # Build model.
    ###########################################################################
    parameters = load("/home/bbdc/tmp/imagenet_224_vgg_16.npz")
    rectify = lasagne.nonlinearities.rectify
    vgg16 = nn_patterns.utils.tests.networks.imagenet.vgg16(rectify)
    lasagne.layers.set_all_param_values(vgg16["out"], parameters)

    # Create prediction model.
    predict_f = theano.function([vgg16["input_var"]],
                                lasagne.layers.get_output(vgg16["out"],
                                                          deterministic=True))


    ###########################################################################
    # Explanations.
    ###########################################################################
    # Create explainers.
    patterns = loadp("/home/bbdc/tmp/imagenet_224_vgg_16.patterns.npz")
    explainers = {}
    for method in methods:
        explainers[method] = nn_patterns.create_explainer(method,
                                                          vgg16["out"],
                                                          patterns=[patterns])

    # Create explanations.
    explanations = np.zeros([len(images), len(explainers), 3, 224, 224])
    text = []
    for i, (image, y) in enumerate(images):
        # Predict label.
        x = eutils.preprocess(image)[None, :, :, :]
        prob = predict_f(x)[0]
        y_hat = prob.argmax()

        text.append((r"\textbf{%s}" % label_to_class_name[y],
                     r"\textit{(%.2f)}" % prob.max(),
                     r"\textit{%s}" % label_to_class_name[y_hat]))

        for eid in explainers:
            # Explain.
            e = explainers[eid].explain(x, target=target)[0]
            # Postprocess.
            e = methods[eid][0](e)
            explanations[i, methods[eid][-1][1]] = e


    ###########################################################################
    # Plot the explanations.
    ###########################################################################

    n_samples = len(images)
    n_padding = n_samples-1
    per_image = 3.2
    shape_per_image = [s + n_padding for s in (224, 224)]
    big_image = np.ones((3,
                         n_padding + n_samples * shape_per_image[1],
                         n_padding + (3+len(methods)) * shape_per_image[0]),
                        dtype=np.float32)

    for i, _ in enumerate(images):
        for eid in explainers:
            egr_idx, e_idx = methods[eid][-1]
            big_image = eutils.put_into_big_image(explanations[i, e_idx],
                                                  big_image, i,
                                                  e_idx + egr_idx,
                                                  n_padding)

    group_fontsize = 20
    fontsize = 15
    plt.figure(figsize=(n_samples * per_image,
                        (3 + len(methods)) * per_image),
               dpi=224)
    plt.clf()
    plt.imshow(big_image.transpose(1, 2, 0), interpolation="nearest")
    plt.tick_params(axis="x", which="both",
                    bottom="off", top="off", labelbottom="off")
    plt.tick_params(axis="y", which="both",
                    bottom="off", top="off", labelbottom="off")
    plt.axis("off")
    plt.rc("text", usetex=True)
    plt.rc("font", family="sans-serif")

    # Plot the labels and probability.
    for i, s_list in enumerate(text):
        for s, offset in zip(s_list, [-50, 0, 50]):
            plt.text(-120,
                     (offset + n_padding + shape_per_image[0]
                      // 2 + shape_per_image[0] * i),
                     s, fontsize=fontsize, ha="center")

    # Plot the methods names.
    for eid in methods:
        egr_idx, e_idx = methods[eid][-1]
        s1, s2 = methods[eid][1]
        plt.text((n_padding + shape_per_image[0] // 2
                  + shape_per_image[0] * (e_idx+egr_idx)),
                 -70, s1, fontsize=fontsize, ha="center")
        plt.text((n_padding + shape_per_image[0] // 2
                  + shape_per_image[0] * (e_idx+egr_idx)),
                 -20, s2, fontsize=fontsize, ha="center")

    # Plot group titles.
    for txt, loc in [("function", 5), ("signal", 11), ("interaction", 18)]:
        plt.text(loc * shape_per_image[0] // 2, -160,
                 r"\textbf{%s}" % txt, fontsize=group_fontsize, ha="center",
                 va="center", color="gray")

    plt.savefig("all_methods.pdf")
