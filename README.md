## FFTW
This package is a wrapper for 1D full and half complex DFT's provided by the FFTW library. Beware, this package is still under some heavy development and interface is subject to change.

## Example
```Go
package main

import (
	"fmt"
	"github.com/bemasher/fftw"
)

func main() {
	dft := fftw.NewDFT1D(8, fftw.Forward, fftw.OutOfPlace, fftw.Measure)
	defer dft.Close()

	// Modifying the location of the input or output arrays will cause FFTW to
	// fail. Instead, copy data into and out of arrays.
	copy(dft.In, []complex128{1, 1, 1, 1, 0, 0, 0, 0})

	dft.Execute()

	fmt.Println(dft)
	fmt.Printf("%+0.3f\n", dft.In)
	fmt.Printf("%+0.3f\n", dft.Out)
}
```
Output:
```
{Kind:C2C Locality:OutOfPlace PlanFlags:NA In:[8]complex128 Out:[8]complex128}
[(+1.000+0.000i) (+1.000+0.000i) (+1.000+0.000i) (+1.000+0.000i) (+0.000+0.000i) (+0.000+0.000i) (+0.000+0.000i) (+0.000+0.000i)]
[(+4.000+0.000i) (+1.000-2.414i) (+0.000+0.000i) (+1.000-0.414i) (+0.000+0.000i) (+1.000+0.414i) (+0.000+0.000i) (+1.000+2.414i)]
```

## Windows
On Windows the library file `libfftw3-3.a` must be placed in the source directory of the package and can be generated using:
	
	dlltool -d libfftw3-3.def -D libfftw3-3.dll -l libfftw3-3.a

## Copyright
The FFTW header file `fftw3.h` carries with the following copyright notice:

	Copyright (c) 2003, 2007-11 Matteo Frigo
	Copyright (c) 2003, 2007-11 Massachusetts Institute of Technology

	The following statement of license applies *only* to this header file,
	and *not* to the other files distributed with FFTW or derived therefrom:

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:

	1. Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
	OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
	ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
	GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
	INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
	WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
