import cheetah
import torch

# Load lattice
ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")

print("Element types:")
print(f"AREAMQZM1: {type(ares_ea.AREAMQZM1)}")
print(f"AREAMQZM2: {type(ares_ea.AREAMQZM2)}")
print(f"AREAMQZM3: {type(ares_ea.AREAMQZM3)}")

print("\nElement attributes:")
print(f"AREAMQZM1 attributes: {dir(ares_ea.AREAMQZM1)}")

print("\nChecking if misalignment is a tracked attribute:")
print(f"Has 'misalignment': {hasattr(ares_ea.AREAMQZM1, 'misalignment')}")

# Check Cheetah version
import cheetah
print(f"\nCheetah version: {cheetah.__version__}")