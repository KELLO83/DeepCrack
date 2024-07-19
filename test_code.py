import torch

# 예시 pred_mask와 mask 텐서
pred_mask = torch.tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]], [[1, 1], [1, 0]], [[0, 0], [0, 1]]])
mask = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]])

# 양성 클래스 정확도 계산

print(mask > 0)
print(pred_mask[mask > 0])
print(pred_mask[mask > 0].eq(mask[mask > 0]))

mask_pos_acc = pred_mask[mask > 0].eq(mask[mask > 0]).sum().item() / mask[mask > 0].numel()

print(f"Positive Class Accuracy: {mask_pos_acc:.4f}")
