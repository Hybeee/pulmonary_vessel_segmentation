from skimage.morphology import skeletonize, thin
from skimage import data
import sknw
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from utils.scan_plotter import scan_to_np_array, view_scan

def old_main_code():
    print("Old main!")
    # HiPaS github vagy sajat megoldas

    # TS a felvetelre -> szivsegment -> + feltetelezes hogy sziv kozeleben adott az er/vena kezdeete
    # skeletonizacio a vena/arteria dolgokra
    # TS-bol megvan a tudo
    # kell a sziv kozeli gyokere a skeletonoknak
    # kivagas szelet hol metszi kozepvonal -> ementen lepunk tovabb

    # img = Image.open('code/images/dogs.jpg')
    # img = np.array(img)

    # img = data.horse()

    ct = scan_to_np_array("C:\\BME\\mester\\1_felev\\onlab_1\\dataset\\HiPaS\\ct_scan_nii\\005.nii")
    img = scan_to_np_array("C:\\BME\\mester\\1_felev\\onlab_1\\dataset\\HiPaS\\annotation\\artery_nii\\005.nii")
    img = scan_to_np_array('dataset/HiPaS/annotation/vein_nii/005.nii')
    print(img.shape)

    img = img[:, :, 143]

    ske = skeletonize(img).astype(np.uint16) # Horse image needs to be inverted(~img), scan image/mask does not -> foreground has to be 1 (white), background has to be 0 (black).

    graph = sknw.build_sknw(ske)

    plt.imshow(img, cmap='gray')

    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')

    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    plt.title('Build graph')
    plt.show()

def test_sknw():
    # img = Image.open('code/images/dogs.jpg')
    # img = np.array(img)

    img = data.horse()
    ske = skeletonize(~img).astype(np.uint16)
    graph = sknw.build_sknw(ske)

    nodes = graph.nodes()
    edges = graph.edges()

    relevant_edges = graph[0][2]['pts']

    for i in graph.edges():
        print(i)

    # for (start, end) in graph.edges():
    #     if (start == 0 and end == 2) or (start == 2 and end == 0):
    #         points = graph[start][end]['pts']
    #         print(points)

    plt.imshow(img, cmap='gray')
    plt.scatter(nodes[0]['o'][1], nodes[0]['o'][0], color="tab:purple", s=100)
    plt.scatter(nodes[2]['o'][1], nodes[2]['o'][0], color="tab:purple", s=100)
    plt.scatter(relevant_edges[:, 1], relevant_edges[:, 0], color="tab:orange", s=5)

    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')

    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    plt.title('Build graph')
    plt.show()

def save_skeleton(image, name):
    skeleton = skeletonize(image).astype(np.uint16)

    skeleton_ske = sitk.GetImageFromArray(skeleton)

    sitk.WriteImage(skeleton_ske, f"dataset/skeleton/{name}_skeleton.nii.gz")

def get_skeleton(image):
    return skeletonize(image).astype(np.uint16)

def view_mask_skeleton(ct, vessel_mask):
    vessel_mask_ske = skeletonize(vessel_mask).astype(np.uint16)

    view_scan([vessel_mask, vessel_mask_ske])


def test_vessel_skeleton():
    vessel_skeleton = scan_to_np_array('dataset/skeleton/005_vein_mask_skeleton.nii.gz')
    graph = sknw.build_sknw(vessel_skeleton)

    # for i in graph.edges():
    #     print(i)

    relevant_edges = graph[0][7]['pts']
    print(relevant_edges)

def main():
    # vessel_mask = scan_to_np_array('dataset/HiPaS/annotation/vein_nii/005.nii')

    # view_mask_skeleton(vessel_mask, vessel_mask)

    # save_skeleton(vessel_mask, "005_vein_mask")

    # skeleton = save_skeleton(image=vessel_mask, name="005_vein_mask_skeleton.nii.gz")

    # print(np.unique(skeleton))

    # test_sknw()
    test_vessel_skeleton()

if __name__ == "__main__":
    main()