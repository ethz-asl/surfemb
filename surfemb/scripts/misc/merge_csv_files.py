import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--csv-paths', nargs='+', required=True)
parser.add_argument('--output-filename', type=str, required=True)

args = parser.parse_args()

csv_paths = args.csv_paths
output_filename = args.output_filename

assert (not os.path.exists(output_filename))

scene_ids = []
view_ids = []
obj_ids = []
scores = []
Rs = []
ts = []
times = []

times_by_scene_ids_and_view_ids = {}

for csv_file_idx, csv_path in enumerate(csv_paths):
    with open(csv_path, "r") as f:
        csv_reader = csv.DictReader(f, fieldnames=[])
        for row_idx, row in enumerate(csv_reader):
            row = row[None]
            if (csv_file_idx == 0):
                scene_ids.append(row[0])
                view_ids.append(row[1])
                obj_ids.append(row[2])
                scores.append(row[3])
                Rs.append(row[4])
                ts.append(row[5])
                times.append(row[6])
            else:
                if (row[3] != "-inf"):
                    assert (scores[row_idx] == "-inf")
                    # Add the result from the current model to the global
                    # results.
                    scene_ids[row_idx] = row[0]
                    view_ids[row_idx] = row[1]
                    obj_ids[row_idx] = row[2]
                    scores[row_idx] = row[3]
                    Rs[row_idx] = row[4]
                    ts[row_idx] = row[5]
                    times[row_idx] = row[6]

            # To avoid evaluation errors, use the same time for each image.
            if ((row[0], row[1]) not in times_by_scene_ids_and_view_ids):
                times_by_scene_ids_and_view_ids[(row[0], row[1])] = row[6]
            times[row_idx] = times_by_scene_ids_and_view_ids[(row[0], row[1])]

        # Check that all files have the same length.
        assert (csv_reader.line_num == len(scores))

for score in scores:
    if (score == "-inf"):
        assert (False)

lines = []
for i in range(len(scores)):
    line = ','.join((
        scene_ids[i],
        view_ids[i],
        obj_ids[i],
        scores[i],
        Rs[i],
        ts[i],
        f'{times[i]}\n',
    ))
    lines.append(line)

with open(args.output_filename, "w") as f:
    f.writelines(lines)
