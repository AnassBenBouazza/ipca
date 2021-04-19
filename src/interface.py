import numpy as np
import requests
import psana
from psana import *

class Detector:
    
    """
    Detector object useful for reshaping data from XTC into detector shape. Stripped
    down version of Detector class from pysingfel when that library isn't available.
    """
    
    def __init__(self, geom_file):
        """
        Initialize instance of Detector class.
        
        :param geom_file: path to *-end.data geometry file
        """
        
        from PSCalib.GeometryAccess import GeometryAccess
        self._geometry = GeometryAccess(geom_file, 0)
        self._compute_pixel_index_map()
        
    def _compute_pixel_index_map(self):
        """
        Compute various parameters related to detector dimensions and pixel indices
        and store as class variables.
        """
        
        # Set coordinate in real space (convert to m)
        temp = [np.asarray(t) * 1e-6 for t in self._geometry.get_pixel_coords()]
        temp_index = [np.asarray(t) for t in self._geometry.get_pixel_coord_indexes()]
        self._panel_num = np.prod(temp[0].shape[:-2])
        self._shape = (self._panel_num, temp[0].shape[-2], temp[0].shape[-1])
        
        pixel_index_map = np.zeros(self._shape + (2,))
        for n in range(2):
            pixel_index_map[..., n] = temp_index[n].reshape(self._shape)
        self._pixel_index_map = pixel_index_map.astype(np.int64)

        self._detector_pixel_num_x = int(np.max(self._pixel_index_map[..., 0]) + 1)
        self._detector_pixel_num_y = int(np.max(self._pixel_index_map[..., 1]) + 1)
        
        self._panel_index = np.zeros((self._detector_pixel_num_x, self._detector_pixel_num_y))
        for l in range(self._panel_num):
            self._panel_index[self._pixel_index_map[l, :, :, 0],
                              self._pixel_index_map[l, :, :, 1]] = l+1
        
        return        
        
    def assemble_image_stack(self, data, dtype=np.int64):
        """
        Reassemble data retrieved from XTC file into shape of detector.
        
        :param data: list of stacked data from each quad of panels
        :param dtype: type of the returned image
        :return image: data in shape of detector
        """
        data = np.array(data).reshape(self._shape)
        image = np.zeros((self._detector_pixel_num_x, self._detector_pixel_num_y)).astype(dtype)
        
        for l in range(self._panel_num):
            image[self._pixel_index_map[l, :, :, 0],
                  self._pixel_index_map[l, :, :, 1]] = data[l, :, :]
            
        return image


class Run:

    """
    Class for retrieving events / shots from a run and applying corrections,
    including pedestal subtraction and masking of bad pixels.
    """

    def __init__(self, expt, run_idx, det=None, image_type=None):
        """
        Initialize instance of Run class.

        :param expt: experiment name
        :param run_idx: run number to load
        :param det: Detector object
        :param image_type: numpy array type
        """
        if det is None:
            raise ValueError('det object needs to be instantiated first')
        if image_type is None:
            self._image_type = np.int64
        else:
            self._image_type = image_type
        self._expt= expt
        self._idx = run_idx
        self._dark_idx = self._retrieve_closest_dark()
        self._det = det
        self._ds  = psana.DataSource(f'exp={expt}:run={run_idx}:smd')
        self._n_images, self._images_nx, self._images_ny, self._image_type = self._retrieve_dimensions()
         
        print(f'{self._n_images} images of shape {self._images_nx}x{self._images_ny} and type {self._image_type}')
        memsize=self._n_images*self._images_nx*self._images_ny*self._image_type.itemsize/1e9
        print(f'Estimated size: {memsize:.2f} GB')

    def _reset_ds(self):
        """
        Reset psana.DataSource object.
        """
        self._ds = psana.DataSource(f'exp={self._expt}:run={self._idx}:smd')

    def _retrieve_dimensions(self):
        """
        Retrieve dimensions of run.

        :return num: number of shots in run
        :return images_nx: length of detector along X axis
        :return images_ny: length of detector along Y axis
        :return image_type: numpy data type for loading shots
        """
        for num,evt in enumerate(self._ds.events()):
            if num == 0:
                data  = self._retrieve_cspad_evt_data(evt)
                image = self._det.assemble_image_stack(data, dtype=self._image_type)
        return num, image.shape[0], image.shape[1], image.dtype

    def _retrieve_cspad_evt_data(self,evt):
        """
        Retrieve intensities collected on CsPad from input event.
    
        :param evt: psana event object
        :return data: list of quads that make up measured data
        """
        data = list()
        cspad = evt.get(psana.CsPad.DataV2, psana.Source('DscCsPad'))
        for num in range(cspad.quads_shape()[0]):
            data.append(cspad.quads(num).data())
        return data

    def _retrieve_closest_dark(self):
        """
        Get the index of the dark run preceding run of interest.

        :return dark_idx: index of relevant dark run
        """
        darks = self._retrieve_dark_list()
        return darks[np.where(darks<self._idx)[0][-1]]
        
    def _retrieve_dark_list(self):
        """
        Get the indices of all dark runs for this experiment.

        :return darks: array of dark run indices
        """
        darks = requests.get(f'https://pswww.slac.stanford.edu/ws/lgbk/lgbk/{self._expt}/ws/get_runs_with_tag?tag=DARK').json()
        if darks['success']:
            #print(f"Dark runs: {darks['value']}")
            return np.array(darks['value'])
        else:
            raise ValueError(f'darks could not be found for {self._expt}')

    def set_dark_idx(self, idx):
        """
        Override index of dark run used for pedestal subtraction.

        :param idx: index of desired dark run 
        """
        self._dark_idx = idx

    def _retrieve_pedestal(self):
        """
        Retrieve the pedestal file associated with the dark run.

        :return pedestal: pedestal array in shape of detector
        """
        pedestal_file = f'/cds/data/psdm/cxi/{self._expt}/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/pedestals/{self._dark_idx}-end.data'
        return self._det.assemble_image_stack(np.loadtxt(pedestal_file))

    def _retrieve_pixel_status(self):
        """
        Retrieve mask corresponding to pixel status, where 0 indicates a good pixel.
        
        :return pixel_status: pedestal array in shape of detector
        """
        pixel_status_file = f'/cds/data/psdm/cxi/{self._expt}/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/pixel_status/{self._dark_idx}-end.data'
        return self._det.assemble_image_stack(np.loadtxt(pixel_status_file))

    def _retrieve_image_batch(self, batch_id=0, batch_size=100, apply_pedestal_correction=True, mask_bad_pixels=True):
        """
        Grab batch of sequential images from run.

        :param batch_id: index of image batch to retrieve 
        :param batch_size: number of images to retrieve
        :param apply_pedestal_correction: boolean that determines whether to pedestal-subract images
        :param mask_bad_pixels: boolean that determines whether to apply the pixel_status mask
        :return: array of images in shape (batch_size, self._images_nx, self._images_ny)
        """
        if batch_id*batch_size > self._n_images:
            raise ValueError('batch is outside dataset')
        if apply_pedestal_correction:
            self._pedestal = self._retrieve_pedestal()
        if mask_bad_pixels:
            self._pixel_status = self._retrieve_pixel_status()
        image_batch = np.zeros((batch_size, self._images_nx, self._images_ny), dtype=self._image_type)

        self._reset_ds()
        i_batch=0
        for num, evt in enumerate(self._ds.events()):
            if num>=batch_id*batch_size and num<(batch_id+1)*batch_size:
                data  = self._retrieve_cspad_evt_data(evt)
                image_batch[i_batch,...] = self._det.assemble_image_stack(data, dtype=self._image_type)
                if apply_pedestal_correction:
                    image_batch[i_batch,...] -= self._pedestal 
                if mask_bad_pixels:
                    image_batch[i_batch,...][self._pixel_status!=0] = 0
                i_batch += 1
        return image_batch
