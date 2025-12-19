import cheetah
import inspect

# Get the source code of Quadrupole.track method
print("=" * 60)
print("Quadrupole.track method signature:")
print("=" * 60)
print(inspect.signature(cheetah.Quadrupole.track))

print("\n" + "=" * 60)
print("Quadrupole.track source code:")
print("=" * 60)
try:
    print(inspect.getsource(cheetah.Quadrupole.track))
except:
    print("Could not retrieve source code")

print("\n" + "=" * 60)
print("Quadrupole.__init__ signature:")
print("=" * 60)
print(inspect.signature(cheetah.Quadrupole.__init__))

print("\n" + "=" * 60)
print("Checking Quadrupole attributes:")
print("=" * 60)
quad = cheetah.Quadrupole(length=0.1, k1=10.0)
print(f"misalignment type: {type(quad.misalignment)}")
print(f"misalignment value: {quad.misalignment}")