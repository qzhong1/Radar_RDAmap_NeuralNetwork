
import h5py 
import numpy as np

class Data_Manager():
    def __init__(self):
        self.nof_frame = 0
        self.nof_range_bin = 256
        self.nof_azm_bin = 32
        self.nof_dop_bin= 384
        self.max_nof_det = 512
        self.skip = 2 # additional fields before frame data

        self.marker = -10

    def init_matrix(self, frame_num):
        self.nof_frame = frame_num
        self.damap_allframe = np.empty(shape=(self.nof_frame, self.nof_range_bin, self.nof_dop_bin, self.nof_azm_bin))
        self.detection_allframe = np.full((self.nof_frame, self.max_nof_det, 6), self.marker)

    def read_damap_single_file(self, h5_file, frame_start, frame_num):
        f = h5py.File(h5_file,'r+')   
        n = f["DAmap"]

        damap_allframe = np.empty(shape=(frame_num, self.nof_range_bin, self.nof_dop_bin, self.nof_azm_bin))
        for i, timeStamp in enumerate(n):
            if i >= frame_start + self.skip and i < frame_start + self.skip + frame_num:
                frame_damap = np.array(n[timeStamp])
                damap_allframe[i-self.skip-frame_start, :, :] = frame_damap

        f.close()
        return damap_allframe

    def read_damap_multi_file(self, h5_dict):
        self.damap_allframe = np.empty(shape=(0, self.nof_range_bin, self.nof_dop_bin, self.nof_azm_bin))
        for h5_file, frame in h5_dict.items():
            damap_allframe = self.read_damap_single_file(h5_file, frame[0], frame[1])
            self.damap_allframe = np.concatenate((self.damap_allframe, damap_allframe), axis=0)


    def read_detection_single_file(self, h5_file, frame_start, frame_num):
        f = h5py.File(h5_file,'r+')   
        frameId = f["rsp_raw_detection"]['frameId']
        frameId = np.squeeze(frameId)
        frameId_uniq = np.unique(frameId)

        nof_acc_det = 0
        nof_cur_det = 0

        detection_allframe = np.full((frame_num, self.max_nof_det, 6), self.marker)
        for i, frameid in enumerate(range(frame_start, frame_start + frame_num)):
            extract_id_start = np.where(frameId == frameId_uniq[frameid])[0][0]
            extract_id_end = np.where(frameId == frameId_uniq[frameid])[0][-1]
            nof_cur_det = extract_id_end - extract_id_start +1

            rsp_raw_detection = f["rsp_raw_detection"]
            range_m = self.extract_data_from_frame(rsp_raw_detection, 'range_m', extract_id_start, extract_id_end)
            doppler_mps = self.extract_data_from_frame(rsp_raw_detection, 'doppler_mps', extract_id_start, extract_id_end)
            azm_bf_rad = self.extract_data_from_frame(rsp_raw_detection, 'azm_bf_rad', extract_id_start, extract_id_end)
            range_bin = self.extract_data_from_frame(rsp_raw_detection, 'range_bin', extract_id_start, extract_id_end)
            raw_doppler_bin = self.extract_data_from_frame(rsp_raw_detection, 'raw_doppler_bin', extract_id_start, extract_id_end)
            azm_bf_bin = self.extract_data_from_frame(rsp_raw_detection, 'azm_bf_bin', extract_id_start, extract_id_end)

            detection_allframe[i, nof_acc_det:nof_cur_det, 0] = range_m
            detection_allframe[i, nof_acc_det:nof_cur_det, 1] = doppler_mps
            detection_allframe[i, nof_acc_det:nof_cur_det, 2] = azm_bf_rad
            detection_allframe[i, nof_acc_det:nof_cur_det, 3] = range_bin
            detection_allframe[i, nof_acc_det:nof_cur_det, 4] = raw_doppler_bin
            detection_allframe[i, nof_acc_det:nof_cur_det, 5] = azm_bf_bin

        f.close()
        return detection_allframe

    def extract_data_from_frame(self, rsp_raw_detection, param_name, startid, endid):
        data = rsp_raw_detection[param_name]
        data = np.squeeze(data)[startid:endid+1]
        return data

    def read_detection_multi_file(self, h5_dict):
        self.detection_allframe = np.full((0, self.max_nof_det, 6), self.marker)
        for h5_file, frame in h5_dict.items():
            detection_allframe = self.read_detection_single_file(h5_file, frame[0], frame[1])
            normalised_data = self.normalise_det(detection_allframe)
            self.detection_allframe = np.concatenate((self.detection_allframe, normalised_data), axis=0)

    def normalise_det(self, det_data):
        nof_frame = det_data.shape[0]
        mask = det_data != self.marker

        means = np.zeros(shape=(nof_frame))
        stds = np.zeros(shape=(nof_frame))

        for frameid in range(nof_frame):
            filtered = det_data[frameid][mask[frameid]]
            means[frameid] = np.mean(filtered)
            stds[frameid] = np.std(filtered)
        stds[stds == 0] = 1
        means = means[:, np.newaxis, np.newaxis]
        stds = stds[:, np.newaxis, np.newaxis]

        normalised_data = (det_data - means) / stds
        normalised_data[det_data == self.marker] = self.marker

        return normalised_data

    # def custom_loss_function(y_true, y_pred, marker_value):
    #     # Create a mask to ignore the special marker values
    #     mask = tf.reduce_all(tf.not_equal(y_true, marker_value), axis=-1)
        
    #     # Apply the mask to the true and predicted values
    #     y_true_masked = tf.boolean_mask(y_true, mask)
    #     y_pred_masked = tf.boolean_mask(y_pred, mask)
        
    #     # Calculate mean squared error for the valid points
    #     mse_loss = K.mean(K.square(y_true_masked - y_pred_masked))
        
    #     return mse_loss


def main():
    h5_data = {"C:\Project\ADC Data\VizTool_RaDar_UDP_2024_6_3_17_44_16_ped\dump_front_left_radar_17_44_16_to_17_44_46_1717407856579_o.h5": [3, 2], 
    "C:\Project\ADC Data\VizTool_RaDar_UDP_2024_6_3_17_44_16_ped\dump_front_left_radar_17_44_16_to_17_44_46_1717407856579_o.h5": [50, 2], 
    "C:\Project\ADC Data\VizTool_RaDar_UDP_2024_6_3_17_44_16_ped\dump_front_left_radar_17_44_16_to_17_44_46_1717407856579_o.h5": [100, 2], 
    "C:\Project\ADC Data\VizTool_RaDar_UDP_2024_6_3_17_44_16_ped\dump_front_left_radar_17_44_46_to_17_45_16_1717407886622_o.h5": [3, 2],
    "C:\Project\ADC Data\VizTool_RaDar_UDP_2024_6_3_17_44_16_ped\dump_front_left_radar_17_44_46_to_17_45_16_1717407886622_o.h5": [50, 2],
    "C:\Project\ADC Data\VizTool_RaDar_UDP_2024_6_3_17_44_16_ped\dump_front_left_radar_17_44_46_to_17_45_16_1717407886622_o.h5": [100, 2]}
    dm = Data_Manager()
    dm.read_damap_multi_file(h5_data)
    dm.read_detection_multi_file(h5_data)
    a = 1


if __name__ == "__main__":
    main()