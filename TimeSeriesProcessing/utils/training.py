import time
import torch

def batch_train(model,
                epoch,
                batch_size,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                set_size,
                device):
    """Train the model in batch
    """
    model.train()  # Turn on the train mode
    batch_loss = 0.
    total_loss = 0.
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        src, tgt_out = batch[0], batch[1]
        src = src.to(torch.float32)
        src = src.to(device)
        tgt_out = tgt_out.to(device)

        optimizer.zero_grad()
        output = model(src)

        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        total_loss += batch_loss
        log_interval = int(set_size / batch_size / 5)
        if i % log_interval == 0 and i > 0:
            cur_loss = batch_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(epoch, i, set_size // batch_size,
                                        scheduler.get_lr()[0],
                                        elapsed * 1000 / log_interval,
                                        cur_loss))
            batch_loss = 0
            start_time = time.time()

    return total_loss


def batch_val(model,
              val_loader,
              criterion,
              device):
    model.eval()
    batch_loss = 0.
    total_loss = 0.
    for batch in val_loader:
        src, tgt_out = batch[0], batch[1]
        src = src.to(torch.float32)
        src = src.to(device)
        tgt_out = tgt_out.to(torch.float32)
        tgt_out = tgt_out.to(device)
        output = model(src)

        loss = criterion(output, tgt_out)
        batch_loss += loss.item()
        total_loss += batch_loss

    return total_loss
