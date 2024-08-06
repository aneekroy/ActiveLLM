import os

TASK = "rte"
MODE = "10"

dir = os.path.dirname(__file__)

# read logs/run/MODE/run_*.log to extract the accuracies and f1 scores
accuracy_list, f1_list = [], []
for filename in os.listdir(os.path.join(dir, "logs/run/" + TASK + "/" + MODE)):
    if filename.endswith(".log"):
        with open(os.path.join(dir, "logs/run/" + TASK + "/" + MODE, filename), "r") as f:
            lines = f.readlines()
            accuracy_list.append(float(lines[-3].split("Acc:")[1]))
            f1_list.append(float(lines[-1].split("F1:")[1]))

print("Acc:", accuracy_list)
print("Acc:", sum(accuracy_list)/len(accuracy_list))
print("F1:", f1_list)
print("F1:", sum(f1_list)/len(f1_list))

# save in logs/run/MODE/avg.txt
with open(os.path.join(dir, "logs/run/" + TASK + "/" + MODE, "avg.txt"), "w") as f:
    f.write("Acc:" + str(accuracy_list) + "\n")
    f.write("Acc:" + str(sum(accuracy_list)/len(accuracy_list)) + "\n")
    f.write("F1:" + str(f1_list) + "\n")
    f.write("F1:" + str(sum(f1_list)/len(f1_list)) + "\n")