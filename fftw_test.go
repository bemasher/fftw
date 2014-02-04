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

func Example_dFT1D() {
	dft := NewDFT1D(8, Forward, OutOfPlace, Measure)
	defer dft.Close()

	// Modifying the location of the input or output arrays will cause FFTW to
	// fail. Instead, copy data into and out of arrays.
	data := []complex128{1, 1, 1, 1, 0, 0, 0, 0}
	copy(dft.In, data)

	dft.Execute()

	fmt.Println(dft)
	fmt.Printf("%+0.3f\n", dft.In)
	fmt.Printf("%+0.3f\n", dft.Out)
	// Output:
	// {Kind:C2C Locality:OutOfPlace PlanFlags:NA In:[8]complex128 Out:[8]complex128}
	// [(+1.000+0.000i) (+1.000+0.000i) (+1.000+0.000i) (+1.000+0.000i) (+0.000+0.000i) (+0.000+0.000i) (+0.000+0.000i) (+0.000+0.000i)]
	// [(+4.000+0.000i) (+1.000-2.414i) (+0.000+0.000i) (+1.000-0.414i) (+0.000+0.000i) (+1.000+0.414i) (+0.000+0.000i) (+1.000+2.414i)]
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

func Example_dFTR2C() {
	r2c := NewDFTR2C(8, OutOfPlace, Measure)
	defer r2c.Close()

	// Modifying the location of the input or output arrays will cause FFTW to
	// fail. Instead, copy data into and out of arrays.
	data := []float64{1, 1, 1, 1, 0, 0, 0, 0}
	copy(r2c.In, data)

	r2c.Execute()

	fmt.Println(r2c)
	fmt.Printf("%+0.3f\n", r2c.In)
	fmt.Printf("%+0.3f\n", r2c.Out)
	// Output:
	// {Kind:R2C Locality:OutOfPlace PlanFlags:NA In:[8]complex128 Out:[5]float64}
	// [+1.000 +1.000 +1.000 +1.000 +0.000 +0.000 +0.000 +0.000]
	// [(+4.000+0.000i) (+1.000-2.414i) (+0.000+0.000i) (+1.000-0.414i) (+0.000+0.000i)]
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

func Example_dFTC2R() {
	c2r := NewDFTC2R(8, OutOfPlace, Measure)
	defer c2r.Close()

	// Modifying the location of the input or output arrays will cause FFTW to
	// fail. Instead, copy data into and out of arrays.
	data := []complex128{(4 + 0i), (1 - 2.414213562373095i), (0 + 0i), (1 - 0.41421356237309515i), (0 + 0i)}
	copy(c2r.In, data)

	c2r.Execute()
	fmt.Println(c2r)
	fmt.Printf("%+0.3f\n", c2r.In)
	fmt.Printf("%+0.3f\n", c2r.Out)
	// Output:
	// {Kind:C2R Locality:OutOfPlace PlanFlags:NA In:[5]complex128 Out:[8]float64}
	// [(+4.000+0.000i) (+1.000-2.414i) (+0.000+0.000i) (+1.000-0.414i) (+0.000+0.000i)]
	// [+8.000 +8.000 +8.000 +8.000 +0.000 +0.000 +0.000 +0.000]
}

const (
	Bits      = 14
	BlockSize = 1 << Bits
)

func BenchmarkDFTInPlace(b *testing.B) {
	dft := NewDFT1D(BlockSize, Forward, InPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkDFTOutOfPlace(b *testing.B) {
	dft := NewDFT1D(BlockSize, Forward, OutOfPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkR2CInPlace(b *testing.B) {
	dft := NewDFTR2C(BlockSize, InPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkR2COutOfPlace(b *testing.B) {
	dft := NewDFTR2C(BlockSize, OutOfPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkC2RInPlace(b *testing.B) {
	dft := NewDFTC2R(BlockSize, InPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkC2ROutOfPlace(b *testing.B) {
	dft := NewDFTC2R(BlockSize, OutOfPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
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
