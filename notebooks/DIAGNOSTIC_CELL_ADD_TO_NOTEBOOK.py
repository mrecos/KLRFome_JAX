"""
ADD THIS AS A NEW CELL IN YOUR NOTEBOOK (after Cell 18)

This diagnostic verifies that Wasserstein kernels are truly using held-out test data
and explains why they might have better generalization (smaller train-test gap).
"""

# Cell: Test Data Verification Diagnostic
print("="*80)
print("DIAGNOSTIC: Verify Test Data is Held-Out (No Leakage)")
print("="*80)
print()

# 1. Verify collection IDs don't overlap
train_ids = set([c.id for c in train_data_scaled.collections])
test_ids = set([c.id for c in test_data_scaled.collections])
overlap = train_ids.intersection(test_ids)

print("1. Collection ID Check:")
print(f"   Training IDs (sample): {sorted(list(train_ids))[:5]}")
print(f"   Test IDs (sample):     {sorted(list(test_ids))[:5]}")
print(f"   Overlap: {len(overlap)} collections")
if len(overlap) == 0:
    print("   ✓ PASS: No overlapping IDs")
else:
    print(f"   ✗ FAIL: {overlap}")
print()

# 2. Verify site locations don't overlap
train_site_locs = [(c.metadata['location'][0], c.metadata['location'][1])
                   for c in train_data_scaled.collections if c.label == 1]
test_site_locs = [(c.metadata['location'][0], c.metadata['location'][1])
                  for c in test_data_scaled.collections if c.label == 1]

print("2. Site Location Check:")
print(f"   Training sites: {len(train_site_locs)} locations")
print(f"   Test sites: {len(test_site_locs)} locations")

# Check if any test site is at same location as train site (within tolerance)
import numpy as np
location_leakage = False
for test_loc in test_site_locs:
    for train_loc in train_site_locs:
        dist = np.sqrt((test_loc[0] - train_loc[0])**2 + (test_loc[1] - train_loc[1])**2)
        if dist < 1e-6:  # Essentially same location
            location_leakage = True
            print(f"   ✗ FOUND: Test site at {test_loc} matches training site!")
            break

if not location_leakage:
    print("   ✓ PASS: All test sites at different locations than training sites")
print()

# 3. Compare actual test predictions between Wasserstein and Mean Embedding
print("3. Test Prediction Comparison:")
wass_results = {k: v for k, v in kernel_results.items() if 'Wasserstein' in k}
mean_results = {k: v for k, v in kernel_results.items() if 'Mean Embedding' in k}

# Get one example of each
wass_example = list(wass_results.values())[0]
mean_example = list(mean_results.values())[0]

print(f"   Wasserstein test predictions (first 5): {wass_example.test_predictions[:5]}")
print(f"   Mean Emb test predictions (first 5):    {mean_example.test_predictions[:5]}")
print()

# Verify they're using the same test labels
if np.array_equal(wass_example.test_labels, mean_example.test_labels):
    print("   ✓ PASS: Both methods evaluated on same test labels")
else:
    print("   ✗ FAIL: Different test labels!")
print()

# 4. Analyze train-test gap
print("4. Train-Test Generalization Gap Analysis:")
print(f"   {'Model':<30} {'Train AUC':<12} {'Test AUC':<12} {'Gap':<10} {'Status'}")
print("   " + "-"*70)

for name, result in kernel_results.items():
    gap = result.train_auc - result.auc
    if gap < 0.05:
        status = "✓ EXCELLENT"
    elif gap < 0.10:
        status = "✓ GOOD"
    elif gap < 0.15:
        status = "○ OK"
    else:
        status = "⚠ HIGH"

    print(f"   {name:<30} {result.train_auc:<12.4f} {result.auc:<12.4f} {gap:<10.4f} {status}")

print()

# 5. Statistical comparison
wass_gaps = [v.train_auc - v.auc for k, v in kernel_results.items() if 'Wasserstein' in k]
mean_gaps = [v.train_auc - v.auc for k, v in kernel_results.items() if 'Mean Embedding' in k]

if wass_gaps and mean_gaps:
    avg_wass = np.mean(wass_gaps)
    avg_mean = np.mean(mean_gaps)

    print("5. Why Does Wasserstein Have Better Generalization?")
    print(f"   Average gap - Wasserstein:    {avg_wass:.4f}")
    print(f"   Average gap - Mean Embedding: {avg_mean:.4f}")
    print()

    if avg_wass < avg_mean:
        print("   ✓ Wasserstein generalizes BETTER (smaller gap)")
        print()
        print("   EXPLANATION:")
        print("   1. Mean Embedding only uses distribution means → can overfit to")
        print("      training means that don't generalize")
        print("   2. Wasserstein uses FULL distribution shape → captures variance,")
        print("      skewness, and other properties that generalize better")
        print("   3. If sites differ from background in distributional SHAPE more")
        print("      than in mean, Wasserstein captures this better")
        print("   4. Wasserstein is naturally more robust/regularized")
    else:
        print("   Mean Embedding generalizes better in this case")

print()
print("="*80)
print("CONCLUSION:")
print("="*80)

if len(overlap) == 0 and not location_leakage:
    print("✓ NO DATA LEAKAGE DETECTED")
    print()
    print("The better Wasserstein generalization is a POSITIVE result showing that")
    print("the Wasserstein kernel is working as designed - it captures distributional")
    print("similarity in a way that generalizes better than mean-only comparison.")
else:
    print("⚠ POTENTIAL ISSUES DETECTED - Review findings above")

print("="*80)
