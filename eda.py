import os
from eda_helper import *


def main():
    # EXTRACT ZIP (IF EXISTS)
    if not extract_zip_if_exists():
        print("Process cancelled.")
        return

    # PATHS
    train_dir = "dataset/seg_train/seg_train"
    test_dir = "dataset/seg_test/seg_test"
    output_dir = "./eda_result"
    ensure_dir(output_dir)


    # CLASS DISTRIBUTION
    train_counts = get_class_distribution(train_dir)
    test_counts = get_class_distribution(test_dir)

    save_text(output_dir, "class_distribution.txt",
            f"Train:\n{train_counts}\n\nTest:\n{test_counts}")

    plot_distribution(train_counts, "Train Distribution",
                    output_dir, "train_distribution.png")

    plot_distribution(test_counts, "Test Distribution",
                    output_dir, "test_distribution.png")


    # SAMPLE IMAGES
    save_samples(train_dir, output_dir, "train_samples.png")
    save_samples(test_dir, output_dir, "test_samples.png")


    # IMAGE SIZE
    widths, heights = analyze_image_sizes(train_dir)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(widths, bins=50)
    plt.title("Width Distribution")
    plt.savefig(os.path.join(output_dir, "width_distribution.png"))
    plt.close()

    plt.figure()
    plt.hist(heights, bins=50)
    plt.title("Height Distribution")
    plt.savefig(os.path.join(output_dir, "height_distribution.png"))
    plt.close()

    size_stats = f"""
    Width: min={min(widths)}, max={max(widths)}
    Height: min={min(heights)}, max={max(heights)}
    """
    save_text(output_dir, "image_size_stats.txt", size_stats)


    # CHANNEL CHECK
    channel_info = check_image_channels(train_dir)

    channel_text = "Image shape distribution:\n"
    for shape, count in channel_info.items():
        channel_text += f"{shape}: {count}\n"

    save_text(output_dir, "channel_distribution.txt", channel_text)


    # CORRUPTED IMAGES
    corrupted_images = find_corrupted_images(train_dir)

    corrupt_text = f"Total corrupted: {len(corrupted_images)}\n"
    corrupt_text += "\n".join(corrupted_images[:50])

    save_text(output_dir, "corrupted_images.txt", corrupt_text)


    # PIXEL DISTRION
    pixel_distribution(train_dir, output_dir, "pixel_distribution.png")
    print(f"EDA complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()