import h5py
import trimesh
import numpy as np

def extract_images():

    db_path = 'hdf5_data/example.hdf5'
    data = h5py.File(db_path, 'r')

    dataset_name = list(data['datasets'].keys())[0]

    print("Dataset: %s" % dataset_name)

    dataset = data['datasets'][dataset_name]

    data_items = list(dataset.keys())

    print("Items: %s" % str(data_items))

    # parse objects data
    objects = dataset['objects']
    print(list(objects.keys()))

    obj = objects['bar_clamp']
    print(list(obj.keys()))

    rendered_images_ds = obj['stable_poses']['pose_1']['rendered_images']

    keys = ['color', 'scaled_depth', 'depth', 'segmask']
    images = { }
    for key in keys:
        images[key] = []
        images_ds = rendered_images_ds[key]
        img_keys = list(images_ds.keys())
        for img_key in img_keys:
            images[key].append(np.asarray(images_ds[img_key]['image_data']))

    return images

if __name__ == "__main__":
    extract_images()

# triangles = np.array(obj_mesh['triangles'])
# vertices = np.array(obj_mesh['vertices'])

# mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
# mesh.show()
