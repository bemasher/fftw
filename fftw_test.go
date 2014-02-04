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

func ExampleDFT1DPlan() {
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

func TestHCDFT1D(t *testing.T) {
	for _, locality := range Localities {
		// Plan forward and backward transforms for current locality
		dftForward := NewHCDFT1D(8, Forward, locality, Measure)
		dftBackward := NewHCDFT1D(8, Backward, locality, Measure)

		// Copy test vector to input and execute
		copy(dftForward.Real, TestInputVectorF64)
		dftForward.Execute()

		// Check output against test output vector
		for i := range dftForward.Complex {
			diff := dftForward.Complex[i] - TestOutputVectorC128[i]
			if math.Abs(real(diff)) > Tolerance || math.Abs(imag(diff)) > Tolerance {
				t.Fatalf("failed: %+0.3f %+0.3f", dftForward.Complex[i], TestOutputVectorC128[i])
			}
		}

		// Copy back to input for backward transform and execute
		copy(dftBackward.Complex, dftForward.Complex)
		dftBackward.Execute()

		// Normalize the output data, FFTW doesn't by default
		for i := range dftBackward.Real {
			dftBackward.Real[i] /= 8
		}

		// Check output against test input vector
		for i := range dftBackward.Real {
			diff := dftBackward.Real[i] - TestInputVectorF64[i]
			if math.Abs(diff) > Tolerance {
				t.Fatalf("failed: %+0.3f %+0.3f", dftBackward.Real[i], TestInputVectorF64[i])
			}
		}
	}
}

func ExampleHCDFT1DPlan() {
	dft := NewHCDFT1D(8, Forward, OutOfPlace, Measure)
	defer dft.Close()

	// Modifying the location of the input or output arrays will cause FFTW to
	// fail. Instead, copy data into and out of arrays.
	copy(dft.Real, []float64{1, 1, 1, 1, 0, 0, 0, 0})

	dft.Execute()

	fmt.Println(dft)
	fmt.Printf("%+0.3f\n", dft.Real)
	fmt.Printf("%+0.3f\n", dft.Complex)
	// Output:
	// {Direction:Forward Locality:OutOfPlace PlanFlags:NA Real:[8]float64 Complex:[5]complex128}
	// [+1.000 +1.000 +1.000 +1.000 +0.000 +0.000 +0.000 +0.000]
	// [(+4.000+0.000i) (+1.000-2.414i) (+0.000+0.000i) (+1.000-0.414i) (+0.000+0.000i)]
}

const (
	Bits      = 14
	BlockSize = 1 << Bits
)

func BenchmarkDFTInPlaceForward(b *testing.B) {
	dft := NewDFT1D(BlockSize, Forward, InPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkDFTOutOfPlaceForward(b *testing.B) {
	dft := NewDFT1D(BlockSize, Forward, OutOfPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkDFTInPlaceBackward(b *testing.B) {
	dft := NewDFT1D(BlockSize, Backward, InPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkDFTOutOfPlaceBackward(b *testing.B) {
	dft := NewDFT1D(BlockSize, Backward, OutOfPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkHCDFTInPlaceForward(b *testing.B) {
	dft := NewHCDFT1D(BlockSize, Forward, InPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkHCDFTOutOfPlaceForward(b *testing.B) {
	dft := NewHCDFT1D(BlockSize, Forward, OutOfPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkHCDFTInPlaceBackward(b *testing.B) {
	dft := NewHCDFT1D(BlockSize, Backward, InPlace, Measure)
	defer dft.Close()

	b.SetBytes(BlockSize)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		dft.Execute()
	}
}

func BenchmarkHCDFTOutOfPlaceBackward(b *testing.B) {
	dft := NewHCDFT1D(BlockSize, Backward, OutOfPlace, Measure)
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
