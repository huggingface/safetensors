The purpose of this directory is to showcase various attacks (and creating your own).


# Torch Arbitrary code execution

Try it out. This will create a seemingly innocuous `torch_ace.pt` file.
```
python torch_ace_create.py
python torch_ace_get_pwned.py
```

# Tensorflow (Keras) Arbitrary Code execution (does not affect `transformers`)

Try it out. This will create a seemingly innocuous `tf_ace.h5` file.
```
python tf_dos_create.py
python tf_dos_get_pwned.py
```

# Torch Denial of Service (OOM kills the running process)

Try it out. This will create a seemingly innocuous `torch_dos.pt` file.
```
python torch_dos_create.py
python torch_dos_get_pwned.py
```

# Numpy Denial of Service (OOM kills the running process)

Try it out. This will create a seemingly innocuous `numpy_dos.npz` file.
```
python numpy_dos_create.py
python numpy_dos_get_pwned.py
```

# Safetensors abuse attempts

In order to try and check the limits, we also try to abuse the current format.
Please ssend ideas !

A few things can be abused:
- Proposal 1: The initial 8 bytes, which could be too big with regards to the file. This crashes, and crashes early (Out of bounds) (Attempt #1).
- Proposal 2: The initial header is JSON, an attacker could use a 4Go JSON file to delay the loads. Debattable how much of an attack this is, but at least 
  it's impossible to "bomb" (like the DOS attacks above) where the files are vastly smaller than their expanded version (because of zip abuse).
  Various "protections" could be put in place, like a header proportion cap (header should always be <<< of the size of the file). (Attempt #2)
- Proposal 3: The offsets could be negative, out of the file. This is all crashing by default.
- Proposal 4: The offsets could overlap. ~~This is actually OK.~~ This is NOT ok.
                While testing Proposal 2, I realized that the tensors themselves where all allocated, and gave me an idea for a DOS exploit where you would have a relatively small
                file a few megs tops, but defining many tensors on the same overlapping part of the file, it was essentially a DOS attack. The mitigation is rather simple, we sanitize the fact
                that the offsets must be contiguous and non overlapping.
- Proposal 5: The offsets could mismatch the declared shapes + dtype. This validated against.
- Proposal 6: The file being mmaped could be modified while it's opened (attacker has access to your filesystem, seems like you're already pwned).
- Proposal 7: serde JSON deserialization abuse (nothing so far: https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=serde). It doesn't mean there isn't a flaw. Same goes for the actual rust compiled binary.

```
python safetensors_abuse_attempt_1.py
python safetensors_abuse_attempt_2.py
python safetensors_abuse_attempt_3.py
```
