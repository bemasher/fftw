package fftw

// #include "fftw3.h"
import "C"

import (
	"strings"
)

// Planning flags provided by FFTW.
type PlanFlag uint

const (
	Measure        PlanFlag = C.FFTW_MEASURE
	DestroyInput   PlanFlag = C.FFTW_DESTROY_INPUT
	Unaligned      PlanFlag = C.FFTW_UNALIGNED
	ConserveMemory PlanFlag = C.FFTW_CONSERVE_MEMORY
	Exhaustive     PlanFlag = C.FFTW_EXHAUSTIVE
	PreserveInput  PlanFlag = C.FFTW_PRESERVE_INPUT
	Patient        PlanFlag = C.FFTW_PATIENT
	Estimate       PlanFlag = C.FFTW_ESTIMATE
	WisdomOnly     PlanFlag = C.FFTW_WISDOM_ONLY
)

func (p PlanFlag) String() (s string) {
	var flags []string
	switch {
	case p&Measure != 0:
		flags = append(flags, "Measure")
	case p&DestroyInput != 0:
		flags = append(flags, "DestroyInput")
	case p&Unaligned != 0:
		flags = append(flags, "Unaligned")
	case p&ConserveMemory != 0:
		flags = append(flags, "ConserveMemory")
	case p&Exhaustive != 0:
		flags = append(flags, "Exhaustive")
	case p&PreserveInput != 0:
		flags = append(flags, "PreserveInput")
	case p&Patient != 0:
		flags = append(flags, "Patient")
	case p&Estimate != 0:
		flags = append(flags, "Estimate")
	case p&WisdomOnly != 0:
		flags = append(flags, "WisdomOnly")
	}
	if len(flags) == 0 {
		return "NA"
	}
	return strings.Join(flags, "|")
}

// Specifies direction of transform, only used for full-complex DFT.
type Direction int

const (
	Forward  Direction = C.FFTW_FORWARD
	Backward Direction = C.FFTW_BACKWARD
)

func (d Direction) String() string {
	switch d {
	case Forward:
		return "Forward"
	case Backward:
		return "Backward"
	default:
		return "Unknown"
	}
}

// Specifies locality of data for transform.
type Locality int

const (
	InPlace Locality = iota
	OutOfPlace
)

func (l Locality) String() string {
	switch l {
	case InPlace:
		return "InPlace"
	case OutOfPlace:
		return "OutOfPlace"
	default:
		return "Unknown"
	}
}
