# Git Worktree Complete Guide & What Just Happened

## 🎯 Summary: What We Accomplished

We just successfully created **two parallel approaches** for experiment orchestration using git worktrees:

1. **Python+YAML approach** (branch: `python-yaml-orchestration`)
2. **Hydra approach** (branch: `hydra-orchestration`) ← **You are here**

Both are **fully committed** and can be compared, tested, and developed independently!

## 🌳 Understanding Git Worktrees

### What Are Worktrees?
Git worktrees let you have **multiple working directories** from the same repository, each potentially on different branches, **simultaneously**.

Think of it as having multiple "checkouts" of your repository at the same time, without the overhead of multiple clones.

### Current Setup After Commits

```bash
git worktree list
```
Shows:
```
/sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm                      b612954 [main]
/sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/hydra-orchestration  6f08141 [hydra-orchestration]
```

## 📊 What Each Worktree Contains

### 1. Main Worktree (`main` branch)
**Location**: `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/locality_tests/`
```
✅ vit_utils.py                    # ViTForProfiling utility
✅ gpu_numa_pipeline_test.py       # Core pipeline (uses ViTForProfiling)
✅ run_gpu_numa_tests.sh          # Original shell orchestration
✅ All existing analysis files
```
**Commit**: `b612954` - "Add ViTForProfiling utility for enhanced D2H transfer testing"

### 2. Python+YAML Branch (`python-yaml-orchestration`)
**Location**: `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/python-yaml-orchestration/locality_tests/`
```
✅ All files from main branch
+ experiment_runner.py             # Custom Python orchestrator
+ configs/                         # YAML configuration files
  ├── numa_locality_study.yaml
  ├── model_scaling_study.yaml
  ├── torch_compile_comparison.yaml
  ├── image_size_study.yaml
  └── quick_test.yaml
+ requirements.txt                 # Dependencies
+ README_experiment_framework.md   # Documentation
```
**Commit**: `aa4fd2b` - "Add Python+YAML experiment orchestration framework"

### 3. Hydra Branch (`hydra-orchestration`) ← **You are here**
**Location**: `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/hydra-orchestration/locality_tests/`
```
✅ All files from main branch
+ hydra_experiment_runner.py       # Hydra-powered orchestrator
+ conf/                            # Hierarchical Hydra configs
  ├── config.yaml                  # Main config
  ├── experiment/
  │   ├── numa_study.yaml
  │   ├── scaling_study.yaml
  │   └── quick_test.yaml
  └── output/
      └── base.yaml
+ requirements_hydra.txt           # Hydra dependencies
+ README_hydra.md                  # Hydra documentation
+ HOW_TO_RUN.md                   # Usage guide
```
**Commit**: `6f08141` - "Add Hydra-based experiment orchestration framework"

## 🔄 What Happened When We Committed?

### Before Commit:
- Hydra worktree had **uncommitted changes**
- Other worktrees were unaffected
- All worktrees shared the same git history up to their branch points

### After Commit:
- **Hydra branch** now has a new commit (`6f08141`)
- **Main branch** is unchanged (`b612954`)  
- **Python+YAML branch** is unchanged (`aa4fd2b`)
- Each worktree can be developed **independently**

### Git History Now Looks Like:
```
b612954 (main) ─┬─ aa4fd2b (python-yaml-orchestration)
                │   Add Python+YAML framework
                │
                └─ 6f08141 (hydra-orchestration) ← You are here
                    Add Hydra framework
```

## 🚀 How to Use This Setup

### 1. Work in Current (Hydra) Worktree
```bash
# You're already here
cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/hydra-orchestration/locality_tests

# Run Hydra experiments
python hydra_experiment_runner.py experiment=quick_test
```

### 2. Switch to Python+YAML Worktree (if needed)
```bash
cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/python-yaml-orchestration/locality_tests

# Run Python+YAML experiments  
python experiment_runner.py configs/quick_test.yaml
```

### 3. Go Back to Original Main Worktree
```bash
cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj-lvm/locality_tests

# Use original shell script
./run_gpu_numa_tests.sh quick_test
```

## 🔍 Key Worktree Commands

### View All Worktrees
```bash
git worktree list
```

### Check Current Branch
```bash
git branch
```

### See Commit History in Current Worktree
```bash
git log --oneline -5
```

### Compare Branches
```bash
# See differences between branches
git log --oneline main..hydra-orchestration
git log --oneline main..python-yaml-orchestration

# See file differences
git diff main..hydra-orchestration
```

## 🎯 Benefits of This Approach

### ✅ **Parallel Development**
- Test both orchestration approaches simultaneously
- No need to switch branches and lose working files
- Independent development without conflicts

### ✅ **Easy Comparison**
- Both approaches use same core pipeline (`gpu_numa_pipeline_test.py`)
- Same ViT configurations available
- Direct performance comparison possible

### ✅ **Risk-Free Experimentation**
- Main branch stays stable
- Each approach can be refined independently
- Easy to abandon one approach if needed

### ✅ **Collaborative Benefits**
- Team members can work on different approaches
- Clear separation of different design philosophies
- Easy to merge best features later

## 🔄 Future Workflow Options

### Option 1: Continue Parallel Development
- Keep both approaches
- Refine each based on usage experience
- Choose winner later based on practical use

### Option 2: Merge Best Features
```bash
# Later, you could merge features from one to the other
git checkout hydra-orchestration
git merge python-yaml-orchestration  # Bring in specific features
```

### Option 3: Choose Winner and Merge to Main
```bash
# If Hydra wins:
git checkout main
git merge hydra-orchestration

# If Python+YAML wins:
git checkout main  
git merge python-yaml-orchestration
```

## 🧹 Cleanup (When Done)

When you're ready to remove worktrees:
```bash
# Remove a worktree (from main worktree)
git worktree remove python-yaml-orchestration

# Delete the branch
git branch -d python-yaml-orchestration
```

## 🎉 Current Status

**You now have:**
1. ✅ **Working Hydra framework** - Ready to run experiments
2. ✅ **Complete documentation** - `HOW_TO_RUN.md` shows you how
3. ✅ **Alternative approach** - Python+YAML available for comparison
4. ✅ **Clean git history** - All changes properly committed
5. ✅ **Parallel development** - Work on either approach independently

**Ready to start experimenting with:**
```bash
python hydra_experiment_runner.py experiment=quick_test
```

The beauty of worktrees is you can now explore both approaches thoroughly without losing any work!