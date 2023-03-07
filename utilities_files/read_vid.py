"""
Ryff 2021 / AI Team
Functions for ROI frame extraction from video for feeding to protobuff or directly to models

____ This Module depends on features specific to Python 3.8 and above ____

"""

import glob
import pickle
from typing import List

import numpy as np
import cv2
import sys
from pathlib import Path

try:
    import dill
except ImportError:
    dill = None
import json
from warnings import warn
from dataclasses import dataclass

try:
    from decord import VideoReader
    from decord import cpu, gpu

    is_decord_available = True
except ModuleNotFoundError:
    is_decord_available = False
    # print("Decord package not found, install to benefit from faster video decoding and read times via GPU")


@dataclass
class VideoData:
    fps: int = 0
    frame_numers: list = None
    n_frames: int = 0
    shot_num: int = 0
    path: str = ""
    n_frames_success = 0
    height: int = 0
    width: int = 0


@dataclass
class ShotSample:
    fps: int = 0
    first_frame_num: int = -1
    last_frame_num: int = -1
    shot_num: int = -1
    first_frames: List[np.ndarray] = None
    last_frame: np.ndarray = None
    prev_shot_last_frame: np.ndarray = None
    next_shot_first_frame: np.ndarray = None
    sample_arrays: List[np.ndarray] = None
    frame_sample_indexes: List[int] = None


import argparse


def parse_args():
    """
    Parse command line arguments if used as standalone
    :return: Parsed argument object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_path', dest='vid_path',
                        help='video path',
                        default='/ml_data/The_Circle/The_Circle_S03/The_Circle_S03E03.mov', type=str)
    parser.add_argument('--sample', dest='do_sample',
                        help='if objective is frame sampling',
                        default='True', type=str)
    parser.add_argument('--preview', dest='show_preview',
                        help='if should open GUI and show a preview of frames',
                        default='', type=str)
    args = parser.parse_args()
    return args


def filter_for_vpp(metadata):
    """

    :param metadata: The metadata json string
    :return: Clean detection metadata where all non-VPP detections are removed
    """

    vpps = []
    for i, hit in enumerate(metadata["hits"]["hits"]):

        for j, detection in enumerate(hit["_source"]["detections"]):

            if detection['type'] == 'ryff.vpp': vpps.append(detection)
    return vpps


def filter_for_seek_shots(metadata):
    """

    :param metadata: The metadata json string
    :return: Clean detection metadata where all non-VPP detections are removed
    """

    shots = []
    for i, hit in enumerate(metadata["hits"]["hits"]):
        shots.append(hit["_source"]["fs"])

    return shots


def get_seek_shots_numbers(video_path):
    """

    :param metadata: The metadata json string
    :return: Clean detection metadata where all non-VPP detections are removed
    """

    metadata = get_metadata(Path(video_path).stem + '.json')

    shots = []
    for i, hit in enumerate(metadata["hits"]["hits"]):
        shots.append(hit["_source"]["fs"])

    return shots


def get_frame_indexes(metadata, unique_frames=True, cluster_indexes_by_shot=True):
    """
    Get all indexes corresponding to detections listed in footage metadata
     as identified between start frame and end frame
     Nb: Returns frames for __all__ detections and needs filtering
     for target hit type (i.e. with filter_for_vpp()) if specific types needed
    :param metadata: A list of detection field metadata objects
    as found at ["hits"]["hits"]["_source"]["detections"]
    :param unique_frames: Bool indicating if detection frames repeated
    from earlier detections are allowed
    :return: Integer list of frame indexes corresponding to detections
    """
    vpp_frame_indexes = []

    for i, hit in enumerate(metadata["hits"]["hits"]):
        # for vpp in metadata:
        fs = hit["_source"]["fs"]
        fe = hit["_source"]["fe"]

        vpp_frame_indexes += [[i for i in range(fs, fe)]] if cluster_indexes_by_shot else [i for i in range(fs, fe)]

    return vpp_frame_indexes if cluster_indexes_by_shot or not unique_frames else set(vpp_frame_indexes)


def get_metadata(filename, parent_dir='/ml_data/metadata'):
    """
    Load metadata json object from disk
    :param filename: Path to metadata file on EC2 instance
    :return: JSON string of metadata
    """

    with open(f'{parent_dir}/{filename}') as f:
        metadata = json.load(f)
    return metadata


def extract_vpps(video_path, group_vpps=False):
    """
    Get all frames corresponding to a footage's VPPs
    :param video_path: Path to video file
    :param group_vpps: Bool where True indicates frames should be returned
    as a seperate list for each VPP. group_vpps=False makes single merged list
     of unique frames containing VPPs
    :return:
    """

    __metadata = get_metadata(Path(video_path).stem + '.json')
    vpp_metadata = filter_for_vpp(__metadata)

    # gets [[VPP 1 indexes],[VPP 2 indexes]] if group_vps and [[Unique indexes]] if don't group
    vpp_frame_indexes = [get_frame_indexes([md], unique_frames=False) for md in vpp_metadata] \
        if group_vpps else [get_frame_indexes(vpp_metadata, unique_frames=True)]

    frames = np.asarray([get_frames(indexes, video_path, dump=False) for indexes in vpp_frame_indexes])

    return frames


def write_samples(vid_path, target_h=216, target_w=384, show_preview=False):
    """
    Write sampled frames to file
    :param vid_path: The footage path
    :param target_h: Height to target for output
    :param target_w: Width to target for output
    :param show_preview: If should show preview in GUI window of extracted frames
    :return:
    """

    get_samples(vid_path, target_h, target_w, True, show_preview)


def get_frames_fast(vid_path=None, uuid=None, frame_nums=None, start_frame=0, end_frame=None, n_frames=1000, thumb=False, source_dir="/ml_data/lower_res/pending", n_proc=4):

    try: from joblib import Parallel, delayed
    except ModuleNotFoundError:
        warn("job lib not found, install first then re-run")
        return None

    assert vid_path or uuid, "must supply either video path or a uuid, got neither"

    if thumb:
        source_dir="/ml_data_fuse/lower_res/micro"

    if not vid_path:
        vid_path = f"{source_dir}/{uuid}.mp4"

    assert Path(vid_path).exists(), f"No file exists at path {vid_path}"

    if frame_nums is not None:

        batches = np.array_split(frame_nums, n_proc)
        r = Parallel(n_jobs=n_proc)(
            delayed(get_frames)(frame_nums=batches[i], vid_path=vid_path) for i in range(n_proc))
    else:

        if not end_frame: end_frame = start_frame + n_frames
        batch_size = (end_frame - start_frame) // n_proc
        r = Parallel(n_jobs=n_proc)(delayed(get_frames_between)(vid_path=vid_path, start_frame=start_frame + (i * batch_size), end_frame=start_frame + ((i + 1) * batch_size)) for i in range(n_proc))

    return np.concatenate(r, axis=0)

def get_frames_between(vid_path=None, uuid=None, start_frame=0, end_frame=None, target_h=216, target_w=384, single_channel=False,
                       resize=False, cap=None, release=True, frames=None, thumb=False, source_dir="/ml_data/lower_res/pending"):

    assert vid_path or uuid, "must supply either video path or a uuid, got neither"

    if thumb:
        source_dir="/ml_data_fuse/lower_res/micro"

    if not vid_path:
        vid_path = f"{source_dir}/{uuid}.mp4"

    assert Path(vid_path).exists(), f"No file exists at path {vid_path}"

    if cap is None:
        cap = cv2.VideoCapture(vid_path)


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not end_frame: end_frame = total_frames - 1

    n_frames = end_frame - start_frame

    if not resize:
        target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frames is None:
        if single_channel:
            frames = np.zeros([n_frames, target_h, target_w], dtype=np.uint8)  # preallocate for efficiency
        else:
            frames = np.zeros([n_frames, target_h, target_w, 3], dtype=np.uint8)  # preallocate for efficiency

    for i in range(start_frame, end_frame):  # skip first frame
        success = cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        if not success: break

        # ret, frame = cap.read()

        if resize:
            frames[i - start_frame] = cv2.resize(cap.read()[1][:, :, 0] if single_channel else cap.read()[1],
                                                 (target_w, target_h)).astype(np.uint8)
        else:
            frames[i - start_frame] = cap.read()[1][:, :, 0] if single_channel else cap.read()[1]

    if release:
        cap.release()

    if not single_channel:
        frames = frames[:, :, :, ::-1]  # rev order of channels

    return frames


def get_samples_fast(vid_path, target_h=216, target_w=384, max_n_samples=None, n_samples=None,
                     acquire_interval=None, sample_interval_seconds=60, frame_numbers=None, video_data=None):
    if not is_decord_available:
        # print("Decord isn't available for fast decording")
        return

    with open(vid_path, 'rb') as f:
        # vr = VideoReader(f, ctx=gpu(0))
        vr = VideoReader(f, ctx=gpu(0))

    print('video frames -- fast arm:', len(vr))

    total_frames = int(len(vr))

    fps = 27

    if n_samples is not None:

        acquire_interval = total_frames // n_samples

    elif acquire_interval is None:
        acquire_interval = int(fps * sample_interval_seconds) if fps > 0 else 27 * sample_interval_seconds

        if acquire_interval >= total_frames:
            acquire_interval = total_frames // max_n_samples if max_n_samples else 10

    if acquire_interval < 1: acquire_interval = 1

    if frame_numbers is None:
        frame_numbers = [i for i in range(acquire_interval, total_frames, acquire_interval)]
        # print("Frame nums", frame_numbers)

    frames = vr.get_batch(frame_numbers)

    frames = frames.asnumpy()
    frames = np.asarray([cv2.resize(f, (target_w, target_h)) for f in frames])

    if video_data is not None:
        video_data.frame_numbers = frame_numbers
        video_data.n_frames = total_frames
        video_data.fps = fps

    return frames


def get_frame_count(vid_path: str, return_cap_instance=False, video_data: VideoData = None):
    """
    Obtain number of frames in video
    :param vid_path: string path to video
    :return:
    """

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_data is not None:
        video_data.fps = cap.get(cv2.CAP_PROP_FPS)
        video_data.n_frames = total_frames
        video_data.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_data.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not return_cap_instance:
        cap.release()
        return total_frames
    else:
        return total_frames, cap


def get_samples(vid_path, target_h=216, target_w=384, dump=False, show_preview=False, max_n_samples=None,
                n_samples=None, start_frame=None,
                acquire_interval=None, sample_interval_seconds=60, vid_buffer=None, frame_numbers=None, video_data=None,
                allow_decord=False, resize=True):
    """
    Sample the footage once every minute.
    If using X/desktop workspace can display the preview frames
    :param vid_path: Path to the footage to sample
    :param sample_interval_seconds: Interval (in seconds) to acquire frames frames where lower numbers,
    default once ever 60 seconds
    mean more frequent samples
    :return: Frame samples
    """

    if is_decord_available and allow_decord:  # decode via fast arm if decord available
        return get_samples_fast(vid_path, target_h, target_w, max_n_samples, n_samples,
                                acquire_interval, sample_interval_seconds, frame_numbers, video_data)

    cap = cv2.VideoCapture(vid_path)

    if resize:
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if video_h == target_h and video_w == target_w: resize = False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # print("Total frames", total_frames)

    if acquire_interval is not None and n_samples is not None: warn(
        f"Parameter set for both acquire interval ({acquire_interval}) and n_samples ({n_samples}) unexpected behaviour"
        f" likely as acquire_interval is overwritten when n_samples is specified. Use n_samples to specifiy number of"
        f" __evenly spaced__ samples taken over whole video. Use max_n_samples if wanting to limit total number frames for a specific acquire_interval")

    frames = []
    if n_samples is not None:

        acquire_interval = total_frames // n_samples

    elif acquire_interval is None:
        acquire_interval = int(fps * sample_interval_seconds) if fps > 0 else 27 * sample_interval_seconds

        if acquire_interval >= total_frames:
            acquire_interval = total_frames // max_n_samples if max_n_samples else 10

    if acquire_interval < 1: acquire_interval = 1

    if frame_numbers is None: frame_numbers = []

    start_frame = acquire_interval if start_frame is None else start_frame

    n_frames_success = 0
    for i in range(start_frame, total_frames, acquire_interval):  # skip first frame

        success = cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        if not success: break

        ret, frame = cap.read()
        if not ret: break

        if resize:
            frame = cv2.resize(frame, (target_w, target_h))  # todo antialias image when downsizing
        frames += [frame]

        n_frames_success += 1

        frame_numbers += [i]  # returned by reference for selection in a presentation

        if show_preview:

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if max_n_samples is not None and len(frames) >= max_n_samples: break

    cap.release()
    if show_preview: cv2.destroyAllWindows()

    if video_data is not None:
        video_data.frame_numbers = frame_numbers
        video_data.n_frames = total_frames
        video_data.fps = fps
        video_data.n_frames_success = n_frames_success

    return np.asarray(frames)[:, :, :, ::-1]  # reverse channel order so no longer BGR


def get_frames(frame_nums, vid_path, target_h=216, target_w=384, dump=False, show_preview=False, resize=False,
               invert_bgr_to_rgb=True, video_data: VideoData = None):
    """
    Get the frames listed in frame_nums
    If using X/desktop workspace can display the preview frames
    :param frame_nums: [int] integer list of frames to extract
    :param vid_path: Path to the footage to sample
    :return: Frame samples
    """

    cap = cv2.VideoCapture(vid_path)

    if not resize:
        target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sized_frame = np.zeros([target_h, target_w, 3], dtype=np.uint8)
    frames = np.zeros([len(frame_nums), target_h, target_w, 3], dtype=np.uint8)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # frames = []

    n_frames_success = 0
    for i in frame_nums:

        success = cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        if not success or i >= total_frames: continue

        # todo return and resize with same array for efficicency
        ret, frame = cap.read()

        if not ret:
            print(f"Unable to get requested frame {i} (total frames {total_frames})")
            continue

        if resize:
            cv2.resize(frame, (target_w, target_h), sized_frame)
            frames[n_frames_success] = sized_frame
            # frames.append(sized_frame)
        else:
            frames[n_frames_success] = frame
            # frames.append(frame)

        n_frames_success += 1

        if show_preview:

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if dump:
        existing_frames = glob.glob(f"/home/ubuntu/notebooks/frames/{Path(vid_path).stem}*")
        dump_name = "/home/ubuntu/notebooks/frames/" + Path(vid_path).stem + "_" + str(len(existing_frames))

        print(f"Writing {dump_name} (num frames: {len(frames)})", flush=True)
        pickle.dump(frames, open(dump_name, "wb"))
        print("Written", flush=True)

    if n_frames_success != len(frame_nums): print(
        f"Couldn't extract {len(frame_nums) - n_frames_success} frames of {len(frame_nums)}", flush=True)

    cap.release()
    if show_preview: cv2.destroyAllWindows()

    if video_data is not None:
        video_data.frame_numbers = frame_nums
        video_data.n_frames = total_frames
        video_data.fps = fps
        video_data.n_frames_success = n_frames_success

    if invert_bgr_to_rgb:
        return np.asarray(frames)[:n_frames_success, :, :, ::-1]  # reverse channel order so no longer BGR
    else:
        return np.asarray(frames)[:n_frames_success]


def get_sample_frame_for_each_shot(video_path=None,
                                   uuid=None,
                                   video_root="/ml_data/lower_res/pending",
                                   sf_lookup=None,
                                   shot_sample_frame_nums=np.array([0, -1]),
                                   shot_sample_freq: int = None,
                                   # frequency of sampling frames instead of using frame numbers
                                   frame_indices: list = None,
                                   max_frames=None,
                                   # set max_frames to None if not to limit, limit is sepererate for each cluster if return_clusters set
                                   return_clusters=True,  # new feature
                                   resize=False,
                                   return_generator=False,
                                   thumb=False,
                                   fast=True,
                                   shot_indices:List=None,
                                   **kargs):
    frame_indices = frame_indices if isinstance(frame_indices, list) else []

    assert isinstance(video_path, str) or isinstance(uuid, str), "Must supply uuid or video path"

    if uuid is None:
        uuid = Path(video_path).stem
    elif uuid:
        if thumb: video_root = "/ml_data_fuse/lower_res/micro"

        video_path = f"{video_root}/{uuid}.mp4"
        assert Path(video_path).exists(), "Video path couldn't be found from uuid"
    else:
        raise AttributeError("Either uuid or video path needed, none supplied")

    if sf_lookup is None:

        sf_lookup = GetSFLookupFromMetadata(uuid=uuid)

    if isinstance(shot_sample_freq, int):
        __frame_indices = [shot_frame_indices[::shot_sample_freq] if len(shot_frame_indices) > 0 else [0] for
                           s, shot_frame_indices in sf_lookup.items()]

    else:
        if not isinstance(shot_sample_frame_nums, np.ndarray):
            if isinstance(shot_sample_frame_nums, list):
                shot_sample_frame_nums = np.array(shot_sample_frame_nums)
            else:
                shot_sample_frame_nums = np.array([shot_sample_frame_nums])

        __frame_indices = [np.asarray(shot_frame_indices)[shot_sample_frame_nums] if len(
            shot_frame_indices) > shot_sample_frame_nums.max()
                           else np.array([shot_frame_indices[0]]) if len(shot_frame_indices) > 0
        else [0] for s, shot_frame_indices in sf_lookup.items()]


    if fast:
        method = get_frames_fast
    else:
        method = get_frames

    if (len(shot_sample_frame_nums) >1 or shot_sample_freq) and return_clusters: #don't cluster if only one frame each shot
        # limit num indices to max_frames if defined
        if isinstance(max_frames, int): __frame_indices = [s[::len(s) // max_frames + 1] if len(s) > max_frames else s
                                                           for s in __frame_indices]

        frame_indices += __frame_indices
        if shot_indices is not None: shot_indices.extend([i for i, s in enumerate(__frame_indices) for f in s])

        if return_generator:
            print("nb returnning generator ~5x slower overall then returning frames in one go")

            gen = (method(frame_nums=shot_indices, vid_path=video_path, **kargs) for shot_indices in frame_indices)
            return gen
        else:
            print("returned frames flat with frame indices grouped by clusters")
            frame_indices_flat = [f for s in __frame_indices for f in s]
            return method(frame_nums=frame_indices_flat, vid_path=video_path, **kargs)

        # return gen if return_generator else [s for s in gen]

    else:
        frame_indices_flat = [f for s in __frame_indices for f in s]
        if shot_indices is not None: shot_indices.extend([i for i, s in enumerate(__frame_indices) for f in s])

        frame_indices += frame_indices_flat
        return method(frame_nums=frame_indices, vid_path=video_path, **kargs)


def GetFrameBatchGenerator(video_path, acquire_interval=5, start_frame=0, end_frame=None, n_frames=None, batch_size=100,
                           invert_bgr_to_rgb=True, video_data: VideoData = None, resize=False):
    if video_data is None: video_data = VideoData()
    n_tot_video_frames = get_frame_count(video_path, video_data=video_data)

    if end_frame is not None:
        n_frames = (end_frame - start_frame) // acquire_interval
    elif n_frames is None:
        n_frames = n_tot_video_frames // acquire_interval
        end_frame = n_tot_video_frames
    else:
        end_frame = min((acquire_interval * n_frames) + start_frame, n_tot_video_frames)  # n_frames specified

    batch_step_size = acquire_interval * batch_size
    n_frames_success = 0

    for batch_i, batch_start in enumerate(range(start_frame, end_frame, batch_step_size)):

        batch_indices = list(
            range(batch_start, min(batch_start + batch_step_size, n_tot_video_frames, end_frame), acquire_interval))
        if len(batch_indices) == 0: return

        frames = get_frames(frame_nums=batch_indices, vid_path=video_path, video_data=video_data,
                            invert_bgr_to_rgb=invert_bgr_to_rgb, resize=resize)
        n_frames_success += video_data.n_frames_success

        yield batch_indices, frames


def GetFramesForRemainingShot(video_path, start_frame_index, allow_big_shots=False):
    from shot_descriptor.boundary import GetModel

    def GetFirstFrameBoundary(search_start_index):

        boundary_model = GetModel()

        frame_generator = GetFrameBatchGenerator(video_path, acquire_interval=1, start_frame=search_start_index)

        for batch_num, (batch_frame_indices, batch_frames) in enumerate(frame_generator):
            boundaries = boundary_model.run_model(frames=batch_frames)
            if len(boundaries) > 0:
                print(f"Found bound at {boundaries[0] + batch_frame_indices[0]}")
                return boundaries[0] + batch_frame_indices[0]

        return batch_frame_indices[-2]

    boundary = GetFirstFrameBoundary(start_frame_index)

    if boundary - start_frame_index > 250 and not allow_big_shots:
        warn(
            f"Shot has large number of frames {boundary - start_frame_index} set allow_big_shots=True to get all frames")
        return None

    frames = get_frames_between(video_path, start_frame=start_frame_index, end_frame=boundary, single_channel=False,
                                resize=False)
    return frames


def GetSFLookupFromMetadata(uuid: str, out: dict = None) -> dict:
    """
    Derive a shot-to-frames lookup from a valossa cvre file
    :param uuid: UUID of filename to source metadata from
    :return: Dictionary
    """

    metadata = get_metadata(uuid + ".json")
    sf_lookup = dict(
        [(i, list(range(s["fs"], s["fe"] + 1))) for i, s in enumerate(metadata["segmentations"]["detected_shots"])])

    if out is not None: out.update(sf_lookup)

    return sf_lookup


def GetFrameFromShot(video_path, uuid, shot_num, shot_frame_index=3):
    try:
        sf_lookup = GetSFLookupFromMetadata(uuid)
        shot_frame_indices = sf_lookup[shot_num]
        video_frame_index = shot_frame_indices[shot_frame_index] if len(shot_frame_indices) > shot_frame_index else 0
        return get_frames([video_frame_index], video_path, resize=False)

    except FileNotFoundError:
        return None


def GetFramesFromShots(uuid, *shot_nums, shot_frame_index=3, all_frames=False, sample_freq=1, video_dir="/ml_data/lower_res/pending", thumb=False):
    try:
        sf_lookup = GetSFLookupFromMetadata(uuid)

        video_frame_index = [k for s in shot_nums for k in sf_lookup[s][::sample_freq]] if all_frames else [sf_lookup[s][shot_frame_index] if len(sf_lookup[s]) > shot_frame_index else 0 for s in
                             shot_nums]

        if thumb:
            video_dir = "/ml_data_fuse/lower_res/micro"

        video_path = f"{video_dir}/{uuid}.mp4"
        # return get_frames(video_frame_index, video_path, resize=False)
        return get_frames_fast(vid_path=video_path, frame_nums=video_frame_index, thumb=thumb)

    except FileNotFoundError:
        return None

if __name__ == '__main__':
    args = parse_args()

    # do_sample = args.do_sample == 'True'
    # if do_sample:
    #     write_samples(args.vid_path, show_preview=args.show_preview != '')
    advert_boundaries_time = [(13876, 20227), (30689, 37042), (49630, 56013), (69860, 76513), (90119, 96443),
                              (108251, 114575), (125364, 131718), (148201, 154465), (171008, 177842), (191058, 198761)]
    exclude_frames = [f for b in [list(range(b[0], b[1])) for b in advert_boundaries_time] for f in b]

    # RewriteVideo(video_path="/ml_data/Americas_Got_Talent_S16E04_ryffcopy (copy 1).mp4",
    #              exclude_frames=exclude_frames, output_path="/ml_data/Americas_Got_Talent_S16E04_ryffcopy_seamless.mp4")
