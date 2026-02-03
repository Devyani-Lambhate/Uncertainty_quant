import numpy as np

def lidar_to_bev(points, 
                 x_range=(-54, 54),    # meters in front/back of sensor
                 y_range=(-54, 54),    # meters left/right
                 z_range=(-5, 3),      # meters height (for filtering)
                 bev_height=500,       # height of BEV image
                 bev_width=500):       # width of BEV image

    # Filter points within ranges
    mask = np.where(
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )[0]
    points = points[mask]

    # Convert coordinates to BEV image coordinates
    x_img = ((points[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * bev_width).astype(np.int32)
    y_img = ((points[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * bev_height).astype(np.int32)

    # Clip to image bounds
    x_img = np.clip(x_img, 0, bev_width - 1)
    y_img = np.clip(y_img, 0, bev_height - 1)

    # Create empty BEV image channels (height, intensity, density)
    height_map = np.zeros((bev_height, bev_width), dtype=np.float32)
    intensity_map = np.zeros((bev_height, bev_width), dtype=np.float32)
    density_map = np.zeros((bev_height, bev_width), dtype=np.float32)

    # For simplicity, assume points[:,3] is intensity if available
    intensity = points[:, 3] if points.shape[1] > 3 else np.ones(points.shape[0])

    # Populate maps, taking max height and max intensity per cell
    for i in range(points.shape[0]):
        x_coord, y_coord = x_img[i], y_img[i]
        height_map[y_coord, x_coord] = max(height_map[y_coord, x_coord], points[i, 2])
        intensity_map[y_coord, x_coord] = max(intensity_map[y_coord, x_coord], intensity[i])
        density_map[y_coord, x_coord] += 1
    
    # Normalize density map to [0,1]
    density_map = np.clip(density_map / 64.0, 0, 1)

    # Stack to form 3-channel BEV image
    bev_image = np.stack([height_map, intensity_map, density_map], axis=-1)

    # Optionally normalize height and intensity to [0,1] for visualization
    bev_image[:, :, 0] = (bev_image[:, :, 0] - z_range[0]) / (z_range[1] - z_range[0])
    bev_image[:, :, 1] = intensity_map / (intensity_map.max() + 1e-6)

    #print("BEV image shape:", bev_image.shape)

    return bev_image

'''
#load an image from a directory and convert to lidar points
path = '/home/saksham/samsad/mtech-project/v1.0-mini/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.bin'
def load_lidar_points(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)[:, :4]  # x, y, z, intensity
    return points   
points = load_lidar_points(path)
bev_image = lidar_to_bev(points)
# plot bev image
import matplotlib.pyplot as plt
plt.imshow(bev_image[:, :, 1], cmap='gray')
plt.title('BEV Image from LIDAR Points')
plt.axis('off')
plt.savefig('bev_image.png')

''' 