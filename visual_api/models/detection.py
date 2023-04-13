
import yaml
import numpy as np

from .image_model import ImageModel
from ..common import NumericalValue, ListValue, StringValue, softmax



class Detection(ImageModel):
    __model__ = 'Detection'

    def __init__(self, network_info, configuration=None):
        super().__init__(network_info, configuration)
        self._check_io_number(1, 1)
        #self.path_to_labels = configuration['path_to_labels']
        if self.path_to_labels:
            self.labels = self._load_labels(self.path_to_labels)
        self.out_layer_name = self._get_outputs()

    def _load_labels(self, labels_file):
        with open(labels_file) as f:
            labels = []
            try:
                my_dict = yaml.load(f, Loader=yaml.FullLoader)
                labels = my_dict['names']
                labels = list(labels.values())
            except yaml.YAMLError as e:
                print(e)
                self.raise_error('The labels file has incorrect format.')
        return labels

    def _get_outputs(self):
        outputs_set = iter(self.outputs)
        out = next(outputs_set)
        pass


    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('crop')
        parameters.update({
            'topk': NumericalValue(value_type=int, default_value=1, min=1),
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(
                description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter"
            ),
        })
        return parameters

    def postprocess(self, outputs, meta):
        outputs = outputs['output0']
        outputs = np.squeeze(outputs)
        predictions = outputs.T

        conf_thresold = 0.6

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_thresold, :]
        scores = scores[scores > conf_thresold]

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        input_height = meta['resized_shape'][0]
        input_width = meta['resized_shape'][1]
        image_height = meta['original_shape'][0]
        image_width = meta['original_shape'][1]

        boxes = predictions[:, :4]

        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)

        indices = self.nms(boxes, scores, 0.05)

        return zip(self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices])


    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y



