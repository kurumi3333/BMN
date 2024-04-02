import mmcv
from mmaction.apis import init_recognizer, inference_recognizer


def temporal_action_detection(config_file, checkpoint_file, video_feature_path, out_file=None):
    # Initialize the model
    model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

    # Load video features
    video_features = mmcv.load(video_feature_path)

    # Perform inference
    results = inference_recognizer(model, video_features)

    if out_file:
        print(f"Saving results to {out_file}")
        mmcv.dump(results, out_file)
    return results


if __name__ == '__main__':
    # Path to the model's config file
    config_path = 'configs/localization/bmn/bmn_activitynet_feat.py'
    # Path to the model checkpoint
    checkpoint_path = 'checkpoints/bmn_activitynet_ckpt.pth'
    # Path to the video features
    feature_path = 'features/example_video_features.pkl'
    # Path to the output file
    output_path = 'results/example_video_result.pkl'

    results = temporal_action_detection(config_path, checkpoint_path, feature_path, output_path)
    print("Detection results:", results)
