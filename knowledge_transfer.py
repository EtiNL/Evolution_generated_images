import torch
import torch.nn.functional as F

def distill_knowledge(teacher_model, student_model, temperature=1.0):
    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

    for param in teacher_model.parameters():
        param.requires_grad = False

    for _ in range(100):
        student_output = student_model(torch.randn(1, 1, 64, 64))
        teacher_output = teacher_model(torch.randn(1, 1, 64, 64))

        loss = F.kl_div(F.log_softmax(student_output / temperature, dim=1),
                        F.softmax(teacher_output / temperature, dim=1),
                        reduction='batchmean') * (temperature ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
