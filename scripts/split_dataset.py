import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("Tongue Dataset Splitting Tool")
    print("=" * 60)

    data_dir = r'D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue-fixed\datasets'
    list_dir = os.path.join(data_dir, 'list')
    seg_dir = os.path.join(data_dir, 'seg')

    print(f"Data directory: {data_dir}")
    print(f"CSV directory: {list_dir}")
    print(f"Image directory: {seg_dir}")

    print("\nChecking original dataset...")

    all_data = []
    for fold in range(1, 6):
        train_file = os.path.join(list_dir, f'train_fold{fold}.csv')
        val_file = os.path.join(list_dir, f'val_fold{fold}.csv')

        if os.path.exists(train_file):
            df = pd.read_csv(train_file)
            df['fold'] = fold
            df['split'] = 'train'
            all_data.append(df)
            print(f"  Loaded train_fold{fold}.csv: {len(df)} rows")

        if os.path.exists(val_file):
            df = pd.read_csv(val_file)
            df['fold'] = fold
            df['split'] = 'val'
            all_data.append(df)
            print(f"  Loaded val_fold{fold}.csv: {len(df)} rows")

    all_df = pd.concat(all_data, ignore_index=True)

    print("\nChecking data issues...")

    total_samples = len(all_df)
    unique_samples = all_df['image_path'].nunique()
    duplicates = total_samples - unique_samples

    print(f"  Total records: {total_samples}")
    print(f"  Unique images: {unique_samples}")
    print(f"  Duplicate records: {duplicates} ({duplicates / total_samples * 100:.1f}%)")

    if duplicates > 0:
        print("  Warning: Duplicate images found! This is the cause of data leakage.")

    print("\nCreating new dataset split...")

    unique_df = all_df.drop_duplicates(subset='image_path', keep='first')
    print(f"  Unique images after deduplication: {len(unique_df)}")

    test_file = os.path.join(list_dir, 'test.csv')
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        print(f"  Test set: {len(test_df)} samples")

        test_paths = set(test_df['image_path'])
        train_val_paths = set(unique_df['image_path'])
        overlap = test_paths.intersection(train_val_paths)

        if overlap:
            print(f"  Warning: Test set and train/val set have {len(overlap)} overlapping samples")
            unique_df = unique_df[~unique_df['image_path'].isin(overlap)]
            print(f"  After removing overlap: train/val set has {len(unique_df)} samples")

    print("\nSplitting into training and validation sets...")

    print("  Using Liver class as stratification (hardest class)")
    unique_df['stratify_label'] = unique_df['Liver'].astype(str)

    stratify_counts = unique_df['stratify_label'].value_counts()
    print(f"  Stratification label distribution:")
    for label, count in stratify_counts.items():
        print(f"    Liver={label}: {count} samples")

    if len(stratify_counts[stratify_counts < 2]) > 0:
        print("  Warning: Some Liver categories have too few samples, using combination of Liver and Kidney")
        unique_df['stratify_label'] = unique_df.apply(
            lambda row: f"{int(row['Liver'])}_{int(row['Kidney'])}", axis=1
        )
        stratify_counts = unique_df['stratify_label'].value_counts()
        print(f"  New stratification label distribution:")
        for label, count in stratify_counts.items():
            print(f"    Liver_Kidney={label}: {count} samples")

    if len(stratify_counts[stratify_counts < 2]) > 0:
        print("  Warning: Still some categories with too few samples, using random split (non-stratified)")
        stratify_label = None
    else:
        stratify_label = unique_df['stratify_label']

    train_df, val_df = train_test_split(
        unique_df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_label
    )

    train_df = train_df.drop(columns=['stratify_label', 'fold', 'split'])
    val_df = val_df.drop(columns=['stratify_label', 'fold', 'split'])

    print(f"  New training set: {len(train_df)} samples")
    print(f"  New validation set: {len(val_df)} samples")
    print(f"  Test set: {len(test_df)} samples")

    print("\nValidating new split...")

    train_paths = set(train_df['image_path'])
    val_paths = set(val_df['image_path'])
    test_paths = set(test_df['image_path'])

    train_val_overlap = train_paths.intersection(val_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    val_test_overlap = val_paths.intersection(test_paths)

    print(f"  Train vs Val overlap: {len(train_val_overlap)}")
    print(f"  Train vs Test overlap: {len(train_test_overlap)}")
    print(f"  Val vs Test overlap: {len(val_test_overlap)}")

    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("  Perfect! The three datasets are completely independent!")
    else:
        print("  Warning: Still some overlap, need further processing")
        return

    print("\nSaving new dataset split...")

    new_list_dir = os.path.join(data_dir, 'list_new')
    os.makedirs(new_list_dir, exist_ok=True)

    train_file = os.path.join(new_list_dir, 'train.csv')
    val_file = os.path.join(new_list_dir, 'val.csv')
    test_file = os.path.join(new_list_dir, 'test.csv')

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"  Training set saved to: {train_file}")
    print(f"  Validation set saved to: {val_file}")
    print(f"  Test set saved to: {test_file}")

    print("\nNew dataset class distribution statistics:")
    labels = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

    print(f"{'Class':<10} {'Training':<15} {'Validation':<15} {'Test':<15}")
    print("-" * 55)

    for label in labels:
        train_pos = train_df[label].sum()
        train_total = len(train_df)
        train_ratio = train_pos / train_total * 100

        val_pos = val_df[label].sum()
        val_total = len(val_df)
        val_ratio = val_pos / val_total * 100

        test_pos = test_df[label].sum()
        test_total = len(test_df)
        test_ratio = test_pos / test_total * 100

        print(f"{label:<10} {train_pos}/{train_total} ({train_ratio:.1f}%)  "
              f"{val_pos}/{val_total} ({val_ratio:.1f}%)  "
              f"{test_pos}/{test_total} ({test_ratio:.1f}%)")

    print("\n" + "=" * 60)
    print("Dataset splitting completed!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Modify debug_train.py to use the new split")
    print(f"2. Run test: python debug_train.py --mode fast --optim baseline")
    print(f"3. Check if AUC is within reasonable range (0.6-0.8)")

    return new_list_dir



    try:
        result = main()
        if result:
            print(f"\nNew split directory: {result}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()