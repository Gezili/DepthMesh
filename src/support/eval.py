import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from support.obstacle_segmentation import generate_depthmap,generate_segmentation_mask

test_images = [900, 1080, 1600]

def move_test_images_to_folder():

    '''
    Helper function for moving labelled images to
    the correct folder to be labelled.
    '''

    for image in test_images:

        rect_img, orig_pc = generate_depthmap(
            frame=image,
            cam1=0,
            cam2=1,
            plot=False
        )

        #Remove regions where mesh is not being
        #generated

        rect_img[:rect_img.shape[0]-210,...] = 255
        rect_img[rect_img.shape[0]-60:,...] = 255

        rect_img = cv2.cvtColor(rect_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'./test/{image}.png', rect_img)

def evaluate_test_images(test_images, input_dir, output_dir, plot=True):

    print('Beginning evaluation')

    '''
    Evaluation function for test images.
    '''

    results = []

    for image in test_images:

        print(f'Generating prediction for frame {image}...')

        #Get prediction

        pred = generate_segmentation_mask(
            image, input_dir, output_dir,
            plot=True, plot_meshes=False
        )

        #Just in case format is not as expected
        #This should load ground truth masks
        #from the masks folder in src

        try:
            actual = cv2.imread(f'./src/support/masks/{image}.png')
            actual = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print('Error - cannot load ground truth masks. Runtime path is not as expected - will skip eval metrics.')

            pred = pred[pred.shape[0]-210:,...]
            pred = pred[:pred.shape[0]-60, ...]

            if plot:

                plt.title('Predicted Obstacle Mask')
                plt.imshow(pred, cmap='gray')
                plt.xticks([])
                plt.yticks([])

                plt.savefig(output_dir + f'/pred_mask_{image}.png')
                plt.close()

            continue

        print(f'Mask done! Running eval metrics for frame {image}...')

        #Crop to evaluation region
        actual = actual[actual.shape[0]-210:,...]
        actual = actual[:actual.shape[0]-60, ...]

        pred = pred[pred.shape[0]-210:,...]
        pred = pred[:pred.shape[0]-60, ...]

        #Binarize ground truth
        actual[actual < 255] = 0
        actual = actual/255

        if plot:

            plt.subplot(121)
            plt.title('Ground Truth')
            plt.imshow(actual, cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(122)
            plt.title('Predicted Obstacle Mask')
            plt.imshow(pred, cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.savefig(output_dir + f'/pred_vs_ground_truth_mask_{image}.png')

            plt.close()

        confusion_matrix = get_confusion_matrix(actual, pred)
        iou = intersection_over_union(actual, pred)

        results.append([image] + list(confusion_matrix.ravel()) + [iou])

    df = pd.DataFrame(results)
    df.columns = ['Image', 'TP', 'FP', 'FN', 'TN', 'IOU']

    df.to_csv(output_dir + '/eval_results.csv', index=None)

def intersection_over_union(actual, pred):

    overlap = pred*actual
    union = pred + actual

    return np.count_nonzero(overlap)/np.count_nonzero(union)

def get_confusion_matrix(actual, pred):

    mat = confusion_matrix(actual.ravel(), pred.ravel())
    return mat

if __name__ == '__main__':

    move_test_images_to_folder()

    evaluate_test_images(plot=True)
