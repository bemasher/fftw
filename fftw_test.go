package fftw

import (
	"fmt"
	"math"
	"testing"
)

const (
	Tolerance = 1e-15
)

var TestInputVectorF64 []float64 = []float64{1, 1, 1, 1, 0, 0, 0, 0}
var TestInputVectorC128 []complex128 = []complex128{1, 1, 1, 1, 0, 0, 0, 0}

var TestOutputVectorC128 []complex128
var TestOutputVector []uint64 = []uint64{
	0x4010000000000000, 0x0000000000000000,
	0x3FF0000000000000, 0xC003504F333F9DE6,
	0x0000000000000000, 0x0000000000000000,
	0x3FF0000000000000, 0xBFDA827999FCEF34,
	0x0000000000000000, 0x0000000000000000,
	0x3FF0000000000000, 0x3FDA827999FCEF34,
	0x0000000000000000, 0x0000000000000000,
	0x3FF0000000000000, 0x4003504F333F9DE6,
}

var Localities []Locality = []Locality{InPlace, OutOfPlace}

func TestDFT1D(t *testing.T) {
	for _, locality := range Localities {
		// Plan forward and backward transforms for current locality
		dftForward := NewDFT1D(8, Forward, locality, Measure)
		dftBackward := NewDFT1D(8, Backward, locality, Measure)

		// Copy test vector to input and execute
		copy(dftForward.In, TestInputVectorC128)
		dftForward.Execute()

		// Check output against test output vector
		for i := range dftForward.Out {
			diff := dftForward.Out[i] - TestOutputVectorC128[i]
			if math.Abs(real(diff)) > Tolerance || math.Abs(imag(diff)) > Tolerance {
				t.Fatalf("failed: %+0.3f %+0.3f", dftForward.Out[i], TestOutputVectorC128[i])
			}
		}

		// Copy back to input for backward transform and execute
		copy(dftBackward.In, dftForward.Out)
		dftBackward.Execute()

		// Normalize the output data, FFTW doesn't by default
		for i, b := range dftBackward.Out {
			dftBackward.Out[i] = complex(real(b)/8, imag(b)/8)
		}

		// Check output against test input vector
		for i := range dftBackward.Out {
			diff := dftBackward.Out[i] - TestInputVectorC128[i]
			if math.Abs(real(diff)) > Tolerance || math.Abs(imag(diff)) > Tolerance {
				t.Fatalf("failed: %+0.3f %+0.3f", dftBackward.Out[i], TestInputVectorC128[i])
			}
		}
	}
}

func TestR2CDFT(t *testing.T) {
	for _, locality := range Localities {
		// Plan transform for current locality
		r2c := NewDFTR2C(8, locality, Measure)

		// Copy test vector to input and execute
		copy(r2c.In, TestInputVectorF64)
		r2c.Execute()

		// Check output against test output vector
		for i := range r2c.Out {
			diff := r2c.Out[i] - TestOutputVectorC128[i]
			if math.Abs(real(diff)) > Tolerance || math.Abs(imag(diff)) > Tolerance {
				t.Fatalf("%10s: %+0.3f %+0.3f", locality, r2c.Out[i], TestOutputVectorC128[i])
			}
		}
	}
}

func TestC2RDFT(t *testing.T) {
	for _, locality := range Localities {
		// Plan transform for current locality
		c2r := NewDFTC2R(8, locality, Measure)

		// Copy test vector to input and execute
		copy(c2r.In, TestOutputVectorC128)
		c2r.Execute()

		// Normalize the output data, FFTW doesn't by default
		for i := range c2r.Out {
			c2r.Out[i] /= 8
		}

		// Check output against test output vector
		for i := range c2r.Out {
			diff := math.Abs(c2r.Out[i] - TestInputVectorF64[i])
			if diff > Tolerance {
				t.Fatalf("R2C %10s: %+0.3f %+0.3f", locality, c2r.Out[i], TestInputVectorF64[i])
			}
		}
	}
}

func Example_dFT1D() {
	dft := NewDFT1D(8, Forward, OutOfPlace, Measure)

	for i := 0; i < 4; i++ {
		dft.In[i] = 1
	}

	dft.Execute()
	fmt.Printf("%+0.3f\n", dft.In)
	fmt.Printf("%+0.3f\n", dft.Out)
	// [(+1.000+0.000i) (+1.000+0.000i) (+1.000+0.000i) (+1.000+0.000i) (+0.000+0.000i) (+0.000+0.000i) (+0.000+0.000i) (+0.000+0.000i)]
	// [(+4.000+0.000i) (+1.000-2.414i) (+0.000+0.000i) (+1.000-0.414i) (+0.000+0.000i) (+1.000+0.414i) (+0.000+0.000i) (+1.000+2.414i)]
}

func init() {
	// Allocate and initialize test output data vector
	TestOutputVectorC128 = make([]complex128, len(TestOutputVector)>>1)
	for i := range TestOutputVectorC128 {
		TestOutputVectorC128[i] = complex(
			math.Float64frombits(TestOutputVector[i<<1]),
			math.Float64frombits(TestOutputVector[i<<1+1]),
		)
	}
}
