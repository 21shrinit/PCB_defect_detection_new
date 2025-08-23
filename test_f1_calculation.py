#!/usr/bin/env python3
"""
Quick F1 Score Calculation Test
==============================

Test to verify F1 calculation with your reported values.
"""

def calculate_f1(precision, recall):
    """Calculate F1 score from precision and recall."""
    if precision <= 0 or recall <= 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Your reported values
precision = 0.9510
recall = 0.9386

# Calculate F1
f1_manual = calculate_f1(precision, recall)

print("F1 Score Calculation Test")
print("=" * 40)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Expected F1: {f1_manual:.4f}")
print(f"Reported F1: 0.0000")
print("=" * 40)

if f1_manual > 0.94:
    print("Your model performance is EXCELLENT!")
    print("F1 should be ~0.9448, not 0.0000")
    print("This confirms the F1 calculation bug")
else:
    print("Unexpected F1 result")

print("\nThe issue is definitely in the F1 extraction, not your model!")