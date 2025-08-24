Validating experiments/roboflow-pcb-training/RB04_YOLOv8n_EIoU_ECA_Roboflow_640px/weights/best.pt...
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:03<00:00,  1.57it/s]
                   all        488        943      0.973      0.968      0.979      0.549
          missing_hole         81        150      0.996      0.993      0.995      0.574
            mouse_bite         77        131      0.957      0.977      0.973      0.556
          open_circuit         80        161      0.987      0.963      0.979      0.555
                 short         93        186      0.973       0.96      0.976      0.574
                  spur         80        155      0.961      0.953      0.969      0.529
       spurious_copper         77        160      0.966      0.963      0.981      0.505
Speed: 0.2ms preprocess, 1.3ms inference, 0.0ms loss, 1.9ms postprocess per image
Results saved to experiments/roboflow-pcb-training/RB04_YOLOv8n_EIoU_ECA_Roboflow_640px
2025-08-24 14:46:37,470 - INFO - ✅ Training completed in 1978.49 seconds
2025-08-24 14:46:37,471 - INFO - ✅ FIXED experiment completed successfully!
2025-08-24 14:46:37,473 - INFO - 📊 Phase 2: Model complexity analysis...
2025-08-24 14:46:37,474 - INFO - 📊 Measuring model complexity: experiments/roboflow-pcb-training/RB04_YOLOv8n_EIoU_ECA_Roboflow_640px/weights/best.pt
2025-08-24 14:46:37,729 - INFO - ⚡ Phase 3: Inference benchmarking...
2025-08-24 14:46:37,730 - INFO - ⚡ Running inference benchmark: experiments/roboflow-pcb-training/RB04_YOLOv8n_EIoU_ECA_Roboflow_640px/weights/best.pt
2025-08-24 14:46:40,454 - INFO - 🧪 Phase 4: Comprehensive testing...
2025-08-24 14:46:40,463 - INFO - 🧪 Running comprehensive testing: experiments/roboflow-pcb-training/RB04_YOLOv8n_EIoU_ECA_Roboflow_640px/weights/best.pt
WARNING ⚠️ 'save_hybrid' is deprecated and will be removed in in the future.
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
val: Fast image access ✅ (ping: 3.0±5.8 ms, read: 24.9±19.4 MB/s, size: 51.3 KB)
val: Scanning /content/drive/MyDrive/PCB_defect_detection_new/datasets/PCB--Defects-DATASET-2/test/labels... 253 images, 0 backgrounds, 0 corrupt: 100% 253/253 [00:00<00:00, 253.49it/s]
val: New cache created: /content/drive/MyDrive/PCB_defect_detection_new/datasets/PCB--Defects-DATASET-2/test/labels.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 16/16 [00:03<00:00,  4.75it/s]
                   all        253        498      0.964      0.949      0.964      0.545
          missing_hole         44         83          1      0.959       0.99      0.627
            mouse_bite         41         88      0.964      0.919      0.952      0.493
          open_circuit         29         60      0.963      0.933      0.952      0.525
                 short         63        120      0.966      0.953       0.94      0.573
                  spur         36         71      0.968      0.972      0.983      0.564
       spurious_copper         40         76      0.924      0.959       0.97       0.49
Speed: 0.9ms preprocess, 2.8ms inference, 0.0ms loss, 2.8ms postprocess per image
Saving /content/drive/MyDrive/PCB_defect_detection_new/runs/detect/val22/predictions.json...
Results saved to /content/drive/MyDrive/PCB_defect_detection_new/runs/detect/val22

Validating experiments/roboflow-pcb-training/RB01_YOLOv8n_SIoU_ECA_Roboflow_640px/weights/best.pt...
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 4/4 [00:03<00:00,  1.32it/s]
                   all        488        943      0.972      0.958      0.977      0.563
          missing_hole         81        150      0.984      0.987      0.994      0.599
            mouse_bite         77        131      0.958      0.954      0.961       0.57
          open_circuit         80        161      0.994      0.955      0.982      0.556
                 short         93        186      0.967      0.941      0.977      0.568
                  spur         80        155      0.974      0.956       0.97      0.539
       spurious_copper         77        160      0.956      0.956      0.975      0.549
Speed: 0.2ms preprocess, 1.3ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to experiments/roboflow-pcb-training/RB01_YOLOv8n_SIoU_ECA_Roboflow_640px
2025-08-24 15:47:10,292 - INFO - ✅ Training completed in 1934.92 seconds
2025-08-24 15:47:10,292 - INFO - ✅ FIXED experiment completed successfully!
2025-08-24 15:47:10,294 - INFO - 📊 Phase 2: Model complexity analysis...
2025-08-24 15:47:10,294 - INFO - 📊 Measuring model complexity: experiments/roboflow-pcb-training/RB01_YOLOv8n_SIoU_ECA_Roboflow_640px/weights/best.pt
2025-08-24 15:47:10,478 - INFO - ⚡ Phase 3: Inference benchmarking...
2025-08-24 15:47:10,479 - INFO - ⚡ Running inference benchmark: experiments/roboflow-pcb-training/RB01_YOLOv8n_SIoU_ECA_Roboflow_640px/weights/best.pt
2025-08-24 15:47:13,181 - INFO - 🧪 Phase 4: Comprehensive testing...
2025-08-24 15:47:13,190 - INFO - 🧪 Running comprehensive testing: experiments/roboflow-pcb-training/RB01_YOLOv8n_SIoU_ECA_Roboflow_640px/weights/best.pt
WARNING ⚠️ 'save_hybrid' is deprecated and will be removed in in the future.
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
val: Fast image access ✅ (ping: 0.4±0.2 ms, read: 39.0±8.1 MB/s, size: 51.3 KB)
val: Scanning /content/drive/MyDrive/PCB_defect_detection_new/datasets/PCB--Defects-DATASET-2/test/labels.cache... 253 images, 0 backgrounds, 0 corrupt: 100% 253/253 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 16/16 [00:03<00:00,  4.96it/s]
                   all        253        498      0.967      0.952      0.966       0.56
          missing_hole         44         83      0.988      0.988      0.993      0.628
            mouse_bite         41         88      0.974      0.932       0.96      0.519
          open_circuit         29         60      0.973      0.933      0.965      0.568
                 short         63        120      0.974      0.941      0.957      0.564
                  spur         36         71          1       0.97      0.981      0.559
       spurious_copper         40         76      0.892      0.947      0.938      0.519
Speed: 1.0ms preprocess, 2.8ms inference, 0.0ms loss, 2.5ms postprocess per image
Saving /content/drive/MyDrive/PCB_defect_detection_new/runs/detect/val23/predictions.json...
Results saved to /content/drive/MyDrive/PCB_defect_detection_new/runs/detect/val23

Validating experiments/roboflow-pcb-training/RB06_YOLOv8n_SIoU_Roboflow_640px/weights/best.pt...
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 4/4 [00:03<00:00,  1.33it/s]
                   all        488        943       0.97       0.97      0.983      0.547
          missing_hole         81        150      0.991      0.993      0.995      0.591
            mouse_bite         77        131      0.948      0.967      0.969      0.544
          open_circuit         80        161      0.984      0.957      0.982      0.559
                 short         93        186      0.978      0.964      0.983      0.537
                  spur         80        155       0.98      0.966      0.984      0.535
       spurious_copper         77        160       0.94      0.974      0.984      0.517
Speed: 0.2ms preprocess, 1.3ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to experiments/roboflow-pcb-training/RB06_YOLOv8n_SIoU_Roboflow_640px
2025-08-24 16:22:07,303 - INFO - ✅ Training completed in 1933.82 seconds
2025-08-24 16:22:07,304 - INFO - ✅ FIXED experiment completed successfully!
2025-08-24 16:22:07,305 - INFO - 📊 Phase 2: Model complexity analysis...
2025-08-24 16:22:07,306 - INFO - 📊 Measuring model complexity: experiments/roboflow-pcb-training/RB06_YOLOv8n_SIoU_Roboflow_640px/weights/best.pt
2025-08-24 16:22:07,493 - INFO - ⚡ Phase 3: Inference benchmarking...
2025-08-24 16:22:07,494 - INFO - ⚡ Running inference benchmark: experiments/roboflow-pcb-training/RB06_YOLOv8n_SIoU_Roboflow_640px/weights/best.pt
2025-08-24 16:22:10,405 - INFO - 🧪 Phase 4: Comprehensive testing...
2025-08-24 16:22:10,413 - INFO - 🧪 Running comprehensive testing: experiments/roboflow-pcb-training/RB06_YOLOv8n_SIoU_Roboflow_640px/weights/best.pt
WARNING ⚠️ 'save_hybrid' is deprecated and will be removed in in the future.
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
val: Fast image access ✅ (ping: 11.1±23.9 ms, read: 18.9±19.1 MB/s, size: 51.3 KB)
val: Scanning /content/drive/MyDrive/PCB_defect_detection_new/datasets/PCB--Defects-DATASET-2/test/labels.cache... 253 images, 0 backgrounds, 0 corrupt: 100% 253/253 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 16/16 [00:03<00:00,  5.04it/s]
                   all        253        498      0.963      0.959       0.97      0.549
          missing_hole         44         83      0.964      0.979      0.987      0.618
            mouse_bite         41         88      0.974      0.955       0.96      0.509
          open_circuit         29         60      0.982      0.933      0.965      0.557
                 short         63        120       0.98      0.967      0.963      0.526
                  spur         36         71      0.964      0.972      0.983      0.555
       spurious_copper         40         76      0.912      0.947      0.959      0.529
Speed: 0.9ms preprocess, 2.7ms inference, 0.0ms loss, 3.2ms postprocess per image

Validating experiments/roboflow-pcb-training/RB00_YOLOv8n_Baseline_Roboflow_640px/weights/best.pt...
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 4/4 [00:02<00:00,  1.37it/s]
                   all        488        943      0.974      0.966       0.98      0.557
          missing_hole         81        150      0.991      0.993      0.995      0.595
            mouse_bite         77        131      0.953      0.962      0.976      0.556
          open_circuit         80        161      0.994      0.968      0.984      0.564
                 short         93        186      0.957      0.952      0.964      0.553
                  spur         80        155      0.973      0.948      0.974      0.542
       spurious_copper         77        160      0.973      0.975      0.984      0.531
Speed: 0.2ms preprocess, 1.4ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to experiments/roboflow-pcb-training/RB00_YOLOv8n_Baseline_Roboflow_640px
2025-08-24 16:45:18,512 - INFO - ✅ Training completed in 1944.62 seconds
2025-08-24 16:45:18,513 - INFO - ✅ FIXED experiment completed successfully!
2025-08-24 16:45:18,514 - INFO - 📊 Phase 2: Model complexity analysis...
2025-08-24 16:45:18,515 - INFO - 📊 Measuring model complexity: experiments/roboflow-pcb-training/RB00_YOLOv8n_Baseline_Roboflow_640px/weights/best.pt
2025-08-24 16:45:18,702 - INFO - ⚡ Phase 3: Inference benchmarking...
2025-08-24 16:45:18,703 - INFO - ⚡ Running inference benchmark: experiments/roboflow-pcb-training/RB00_YOLOv8n_Baseline_Roboflow_640px/weights/best.pt
2025-08-24 16:45:21,243 - INFO - 🧪 Phase 4: Comprehensive testing...
2025-08-24 16:45:21,251 - INFO - 🧪 Running comprehensive testing: experiments/roboflow-pcb-training/RB00_YOLOv8n_Baseline_Roboflow_640px/weights/best.pt
WARNING ⚠️ 'save_hybrid' is deprecated and will be removed in in the future.
Ultralytics 8.3.179 🚀 Python-3.12.11 torch-2.8.0+cu126 CUDA:0 (NVIDIA L4, 22693MiB)
Model summary (fused): 72 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
val: Fast image access ✅ (ping: 0.3±0.1 ms, read: 36.8±9.2 MB/s, size: 51.3 KB)
val: Scanning /content/drive/MyDrive/PCB_defect_detection_new/datasets/PCB--Defects-DATASET-2/test/labels... 253 images, 0 backgrounds, 0 corrupt: 100% 253/253 [00:01<00:00, 200.40it/s]
val: New cache created: /content/drive/MyDrive/PCB_defect_detection_new/datasets/PCB--Defects-DATASET-2/test/labels.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 16/16 [00:03<00:00,  5.19it/s]
                   all        253        498      0.959      0.951      0.967      0.551
          missing_hole         44         83          1      0.969      0.993      0.639
            mouse_bite         41         88      0.955      0.959      0.966      0.529
          open_circuit         29         60      0.943      0.933      0.952      0.536
                 short         63        120      0.974      0.967      0.963      0.565
                  spur         36         71      0.964      0.972      0.978      0.534
       spurious_copper         40         76       0.92      0.908      0.952      0.503
Speed: 0.7ms preprocess, 3.0ms inference, 0.0ms loss, 2.3ms postprocess per image
Saving /content/drive/MyDrive/PCB_defect_detection_new/runs/detect/val19/predictions.json...
Results saved to /content/drive/MyDrive/PCB_defect_detection_new/runs/detect/val19