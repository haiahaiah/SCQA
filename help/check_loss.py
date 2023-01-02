
with open('loss.txt', 'r') as f:
    total_loss, cnt = 0, 0
    for line in f:
        cnt += 1
        line = line.strip()
        loss = float(line[line.index(":") + 1: line.index(",")])
        total_loss += loss
        print(loss)
    print(total_loss / cnt)
