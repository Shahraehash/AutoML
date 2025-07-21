# Memory Management & Model Storage Guide

## Advanced Memory Management System

MILO-ML includes sophisticated memory management to handle large multi-class problems and prevent system crashes during intensive model training.

### Real-time Memory Monitoring

**Automatic Monitoring Features**
- **Continuous Tracking**: Real-time memory usage monitoring during training
- **Smart Thresholds**: Warning at 75% usage, critical at 85% usage
- **Predictive Alerts**: Early warnings before memory exhaustion
- **Resource Estimation**: Projects memory needs for remaining training

**User Notifications**
```
Memory Status Indicators:
├── Green (0-75%): Normal operation
├── Yellow (75-85%): Warning - cleanup in progress
├── Red (85%+): Critical - emergency protocols active
└── Progress Updates: Memory-aware training progress
```

### Intelligent Cleanup Strategies

**Automatic Memory Management**
- **Progressive Cleanup**: Memory freed after each model completes training
- **Joblib Resource Management**: Specialized cleanup for sklearn/joblib temporary files
- **Emergency Protocols**: Aggressive cleanup when memory becomes critical
- **Process Isolation**: Clean worker processes to prevent memory leaks

**Multi-class Specific Optimizations**
```
Memory Management During Training:
├── Model Training: Individual models trained sequentially
├── Immediate Saving: Models saved to disk immediately after training
├── Memory Recycling: RAM cleared after each model save
├── Archive Creation: Models compressed into space-efficient archives
└── On-Demand Access: Models loaded only when needed for analysis
```

### Memory Usage Patterns

**Typical Training Memory Profile**
```
Training Phase Breakdown:
├── Data Loading: 15% of available memory
├── Single Model Training: 25-35% peak usage
├── Multi-class Training: 35-50% peak usage
├── OvR Re-optimization: 50-70% peak usage (with active cleanup)
├── Results Generation: 20-30% sustained usage
└── Export Operations: 25-40% temporary spike
```

**Resource Scaling by Problem Size**
- **Small Problems** (2-3 classes, <1000 samples): 2-4GB recommended
- **Medium Problems** (4-6 classes, 1000-10000 samples): 8-16GB recommended  
- **Large Problems** (7+ classes, 10000+ samples): 16-32GB recommended
- **OvR Re-optimization**: Add 50-100% to above recommendations

## Efficient Model Storage System

### Progressive Storage Strategy

**During Training Process**
```
Storage Workflow:
├── Model Creation: Models trained in memory
├── Immediate Save: Models saved as individual .joblib files
├── Memory Cleanup: Model removed from RAM after saving
├── Batch Compression: Groups of models compressed into archives
└── Archive Management: Individual files deleted after archiving
```

**Storage Structure**
```
Project Directory:
├── /models/
│   ├── main_models/
│   │   ├── model_key_1.joblib (temporarily)
│   │   ├── model_key_2.joblib (temporarily)
│   │   └── ...
│   ├── ovr_models/
│   │   ├── model_key_1_ovr_class_0.joblib (temporarily)
│   │   ├── model_key_1_ovr_class_1.joblib (temporarily)
│   │   └── ...
│   ├── main_models.tar.gz (final compressed archive)
│   └── ovr_models.tar.gz (final compressed archive)
├── report.csv
├── metadata.json
└── class_results/ (class-specific analysis data)
```

### Compression Benefits

**Storage Efficiency**
- **Space Reduction**: 60-80% smaller storage footprint
- **Faster Loading**: Compressed archives load faster than individual files
- **Organized Access**: Related models grouped logically
- **Bandwidth Efficiency**: Faster transfers and backups

**Performance Metrics**
```
Typical Storage Savings:
├── Individual Files: 2.3GB total storage
├── Compressed Archives: 0.8GB total storage
├── Space Savings: 65% reduction in disk usage
├── Load Performance: 40% faster access times
└── Memory Efficiency: 70% lower peak memory during analysis
```

### On-Demand Model Access

**Smart Loading System**
- **Selective Extraction**: Only requested models extracted from archives
- **Temporary Workspace**: Extracted models placed in temporary directories
- **Automatic Cleanup**: Temporary files removed after analysis complete
- **Cache Management**: Frequently accessed models cached briefly in memory

**Access Patterns**
```
Model Access Workflow:
├── User Request: Selects specific model for analysis/export
├── Archive Check: Determines if model is in compressed archive
├── Smart Extraction: Extracts only the needed model files
├── Temporary Storage: Model placed in temporary workspace
├── Analysis/Export: Model used for requested operation
└── Cleanup: Temporary files automatically removed
```

## Memory Management During Training

### Progressive Training Strategy

**Sequential Model Training**
- **One-at-a-Time**: Models trained individually to control memory usage
- **Immediate Persistence**: Each model saved to disk immediately after training
- **Memory Recycling**: RAM freed completely between models
- **Progress Tracking**: Memory-aware progress reporting

**Multi-class Considerations**
```
Training Sequence for 4-class Problem:
├── Main Multi-class Models: Train all algorithm combinations
├── Memory Check: Verify sufficient resources for OvR (if enabled)
├── OvR Class 0 Models: Train all algorithm combinations for Class 0 vs Rest
├── OvR Class 1 Models: Train all algorithm combinations for Class 1 vs Rest
├── OvR Class 2 Models: Train all algorithm combinations for Class 2 vs Rest
├── OvR Class 3 Models: Train all algorithm combinations for Class 3 vs Rest
└── Archive Creation: Compress all models into final archives
```

### Emergency Memory Management

**Critical Memory Protocols**
- **Early Detection**: System detects memory pressure before exhaustion
- **Aggressive Cleanup**: Enhanced cleanup procedures activated
- **Training Continuation**: Attempts to continue training with reduced memory footprint
- **Graceful Degradation**: If memory cannot be recovered, training stops safely

**User Notifications**
```
Memory Management Alerts:
├── "Memory optimization in progress..." (Normal cleanup)
├── "High memory usage detected, performing cleanup..." (Warning)
├── "Critical memory level, aggressive cleanup active..." (Critical)
└── "Training stopped due to insufficient memory" (Emergency stop)
```

## Best Practices for Memory Management

### Resource Planning

**Before Starting Training**
1. **Estimate Requirements**: Use dataset size and complexity to estimate memory needs
2. **Close Other Applications**: Free up system resources before training
3. **Monitor Available Resources**: Check available RAM and disk space
4. **Choose Training Mode**: Consider resource impact of OvR re-optimization

**During Training**
- **Monitor Progress**: Watch for memory warnings in the interface
- **Avoid Other Tasks**: Don't run memory-intensive applications during training
- **Check Disk Space**: Ensure adequate space for model storage and temporary files

### Optimization Strategies

**For Resource-Constrained Environments**
- **Use Efficient Mode**: Avoid OvR re-optimization to reduce memory usage
- **Reduce Algorithm Selection**: Train fewer algorithm combinations
- **Batch Processing**: Process smaller subsets of the pipeline combinations
- **Increase Virtual Memory**: Configure appropriate swap space

**For Large-Scale Problems**
- **Plan for OvR Overhead**: Account for N× memory usage when re-optimizing
- **Staged Training**: Consider running smaller batches of algorithms
- **Infrastructure Scaling**: Use systems with adequate RAM for problem size
- **Monitor Resource Usage**: Track memory patterns for future planning

## Performance Monitoring

### Real-time Statistics

**Memory Tracking**
```
Live Memory Dashboard:
├── Current Usage: 4.2GB / 16GB (26%)
├── Peak Usage: 6.8GB (43%)
├── Models Completed: 847 / 1200
├── Estimated Remaining: 2.1GB peak needed
└── Status: Normal operation
```

**Storage Tracking**
```
Storage Statistics:
├── Models Saved: 847 individual files
├── Disk Usage: 1.2GB (before compression)
├── Compression Ratio: 68% reduction achieved
├── Archive Status: 3 archives created
└── Cleanup Status: 245 temporary files removed
```

This comprehensive memory management system ensures that even very large multi-class problems can be trained successfully while maintaining system stability and optimal performance.