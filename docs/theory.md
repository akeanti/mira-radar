# FMCW radar from first principles

Everything MIRA does, derived in the order the signal flows through it. Each
section names the file that implements it, so you can read the maths and the
code side by side. Numbers in examples are for the default `MIRA-77` preset.

1. [The chirp](#1-the-chirp)
2. [Range: a delay becomes a frequency](#2-range-a-delay-becomes-a-frequency)
3. [Velocity: the phase across chirps](#3-velocity-the-phase-across-chirps)
4. [Angle: the phase across antennas, and MIMO](#4-angle-the-phase-across-antennas-and-mimo)
5. [How strong is the echo: the radar equation](#5-how-strong-is-the-echo-the-radar-equation)
6. [Detection: CFAR](#6-detection-cfar)
7. [Beyond the beamwidth: MUSIC](#7-beyond-the-beamwidth-music)
8. [Tracking](#8-tracking)
9. [The laptop sonar](#9-the-laptop-sonar)
10. [Further reading](#10-further-reading)

---

## 1. The chirp

An FMCW radar transmits a *chirp*: a sinusoid whose frequency rises linearly,

$$f_{TX}(t) = f_0 + S\,t, \qquad 0 \le t \le T_{ramp},$$

with slope $S$ in Hz/s (30 MHz/µs here). The phase is the integral of the frequency,

$$\phi_{TX}(t) = 2\pi\left(f_0 t + \tfrac{1}{2} S t^2\right).$$

A target at range $R$ sends back a copy delayed by the round trip $\tau = 2R/c$.
At 20 m that is 133 ns, tiny compared with a 32 µs ramp, but it is all we need.

![chirp, beat signal and range FFT](media/fmcw-basics.png)

## 2. Range: a delay becomes a frequency

The receiver multiplies the echo by the transmitted chirp (a mixer) and keeps
the difference frequency. With the echo's phase $\phi_{TX}(t-\tau)$:

$$\phi_{IF}(t) = \phi_{TX}(t) - \phi_{TX}(t-\tau) = 2\pi\left(f_0\tau + S\tau\,t - \tfrac{1}{2}S\tau^2\right).$$

The term in $t$ is a constant frequency, the **beat frequency**:

$$f_b = S\tau = \frac{2SR}{c}.$$

So each target becomes a pure tone whose frequency is proportional to its
range, and an FFT of the ADC samples of one chirp is a range profile
([`range_doppler.py`](../mira/processing/range_doppler.py)). The last term,
$\tfrac12 S\tau^2$, is the *residual video phase*: tiny, but the simulator
includes it anyway.

**Range resolution.** Two tones can be told apart by an FFT of duration $T_s$
(the ADC window) if they differ by at least $1/T_s$. In range:

$$\Delta R = \frac{c}{2 S T_s} = \frac{c}{2B},$$

where $B = S T_s$ is the bandwidth swept *while sampling*. 768 MHz gives 19.5 cm.
It depends on nothing but bandwidth, which is why mmWave radars sweep GHz.

**Maximum range** is set by the IF bandwidth. With complex (I/Q) sampling at
$f_s$, beat frequencies up to $f_s$ fit: $R_{max} = f_s c / 2S$ = 50 m.

## 3. Velocity: the phase across chirps

Fire $N$ chirps back to back, $T_c$ apart. If the target moves at radial
velocity $v$, its range grows by $vT_c$ between chirps. That is far too small
to change the beat frequency, but the constant phase term $2\pi f_0\tau$
changes by

$$\Delta\phi = \frac{4\pi v T_c}{\lambda}.$$

A linear phase ramp across chirps is again a tone, now in *slow time*, at the
Doppler frequency $f_d = 2v/\lambda$. A second FFT across chirps, for every
range bin, gives the **range-Doppler map**.

$$\Delta v = \frac{\lambda}{2 N T_c}, \qquad v_{max} = \pm\frac{\lambda}{4T_c}.$$

With 128 loops of 78 µs at 77.5 GHz: 0.19 m/s and ±12.4 m/s. Faster targets
alias, exactly like a wagon wheel in a film.

**Static clutter removal.** Anything that does not move has the same phase on
every chirp. Subtracting the mean over chirps in each range bin before the
Doppler FFT deletes it completely, walls, posts and the TX→RX leakage alike.
That is a zero-Doppler notch; the classic two-pulse MTI canceller
$y_n - y_{n-1}$ is its little brother (the sonar uses that one).

![range-Doppler map before and after clutter removal](media/range-doppler.png)

## 4. Angle: the phase across antennas, and MIMO

Put receive antennas on a line, $d$ apart. A target at azimuth $\theta$ is a
little closer to one end, by $d\sin\theta$ per antenna, so the phase steps by

$$\Delta\phi = \frac{2\pi d \sin\theta}{\lambda}$$

from one antenna to the next: a third tone, this time in space. With
$d = \lambda/2$, $\sin\theta$ maps onto the full FFT band without ambiguity.
The angular resolution of $N$ antennas is about $2/N$ radians at boresight.

**MIMO.** 4 RX antennas give a 29° beam. Instead of building more receivers,
add transmitters spaced $N_{RX}\cdot d$ apart and fire them one after the other
(*time-division multiplexing*). The phase seen by the pair (TX $m$, RX $n$)
depends on the sum of their positions, so 2 TX × 4 RX behave like 8 receivers
in a row: the **virtual array**, 14° beam, for the price of 6 antennas.

**The catch: TDM motion.** TX 1 fires one chirp period after TX 0, and a
moving target has rotated its phase by $2\pi f_d T_c$ in that time. Read
naively, that extra phase looks like an angle. MIMO radars must undo it with
the Doppler measured on the same detection:

$$x_p \leftarrow x_p\, e^{-j 2\pi f_d\, m(p)\, T_c}.$$

[`tdm_phase_correction`](../mira/processing/angle.py) does this, and
[`test_tdm_mimo_motion_compensation_is_needed`](../tests/test_simulator.py)
shows an 11 m/s target landing several degrees off without it.

The simulator writes all three tones in one expression
([`simulator.py`](../mira/simulator.py)): for scatterer $k$, chirp $l$, virtual
channel $p$ and sample $i$,

$$s[l,p,i] = a_k\,\exp j\Big(2\pi f_{b,k}\,t_i + 2\pi f_{d,k}\,\tau_{l,m(p)} - 2\pi x_p\sin\theta_k + \phi_{0,k}\Big).$$

Each phase term depends on only one index, so the whole frame is a single
matrix product of a (chirps × channels, scatterers) matrix by a (scatterers,
samples) matrix.

## 5. How strong is the echo: the radar equation

Received power from a target of radar cross section $\sigma$ at range $R$:

$$P_r = \frac{P_t\, G_t G_r\, \lambda^2\, \sigma}{(4\pi)^3 R^4}.$$

Noise power per complex sample is $kTF f_s$ (Boltzmann, temperature, noise
figure, bandwidth). Per sample the SNR of a car at 50 m is negative, about
−15 dB. The FFTs rescue it: a tone adds up coherently over the $N_s$ samples,
$N$ chirps and $N_v$ channels while noise adds up in power, a processing gain
of $10\log_{10}(N_s N N_v)$ = 54 dB.
[`test_snr_follows_the_radar_equation`](../tests/test_simulator.py) checks the
simulator against this.

## 6. Detection: CFAR

The noise floor of a range-Doppler map is never flat: clutter residue,
sidelobes of strong targets, range-dependent receiver noise. A fixed threshold
either floods with false alarms or misses weak targets. **Constant false alarm
rate** detection estimates the noise around each cell from a ring of training
cells, skipping guard cells next to the cell under test, and declares a
detection if

$$P_{CUT} > \alpha \cdot \frac{1}{N}\sum_{i=1}^{N} P_i.$$

If every cell is exponentially distributed noise (one channel, square-law
detector), choosing $\alpha$ for a false alarm probability $P_{fa}$ gives the
textbook

$$\alpha = N\left(P_{fa}^{-1/N} - 1\right).$$

MIRA sums $|X|^2$ over 8 virtual channels before CFAR, so each cell is
$\Gamma(8)$, not exponential, and that formula would be wrong. With
$X \sim \Gamma(P)$ for the cell and $Z = \sum P_i \sim \Gamma(NP)$ for the
training cells, $X/(X+Z)$ is Beta distributed, so

$$P_{fa} = 1 - I_{t/(1+t)}(P, NP), \qquad t = \alpha/N,$$

with $I$ the regularized incomplete beta function, solved for $\alpha$ by
bisection ([`ca_cfar_scale`](../mira/processing/cfar.py), with its own
continued-fraction `betainc` so there is no SciPy dependency).

**Ordered-statistic CFAR** replaces the mean by the $k$-th smallest training
cell. A strong neighbour then lands among the largest cells and is ignored, so
it cannot raise the threshold and hide a weak target next to it:

![CA-CFAR vs OS-CFAR](media/cfar.png)

The training sums are computed for every cell at once with a summed-area table,
and the same running-sum idea is what makes the [Verilog core](../hdl/README.md)
cheap.

## 7. Beyond the beamwidth: MUSIC

Beamforming (the angle FFT) cannot separate two targets closer than the
beamwidth. MUSIC splits the covariance matrix of the array into a signal
subspace and a noise subspace and looks for the directions whose steering
vectors $a(\theta)$ are orthogonal to the noise subspace $E_n$:

$$P_{MUSIC}(\theta) = \frac{1}{a(\theta)^H E_n E_n^H a(\theta)}.$$

Two problems on a radar: there is only one snapshot per detection, and the
two echoes are coherent (same chirp), which makes the covariance rank one.
**Forward-backward spatial smoothing** fixes both: average the covariance of
overlapping sub-arrays, and of the conjugate-reversed array. The price is a
smaller effective aperture (6 of 8 elements here).

![MUSIC vs FFT](media/music-vs-fft.png)

## 8. Tracking

A detection list is noisy, flickers, and says nothing about where things are
going. The tracker ([`tracking.py`](../mira/processing/tracking.py)) keeps a
state $\mathbf{x} = [x, y, v_x, v_y]$ per object with a constant-velocity model
and measures, per frame, the centroid of the points assigned to it:

$$h(\mathbf{x}) = \begin{bmatrix} \sqrt{x^2+y^2} \\ \operatorname{atan2}(x, y) \\ (x v_x + y v_y)/\sqrt{x^2+y^2} \end{bmatrix}.$$

The third row, radial velocity, is the radar's superpower: it is measured
directly from Doppler, so the filter knows how fast an object approaches from
its very first frame instead of differentiating noisy positions. $h$ is
non-linear, so the filter is an **extended** Kalman filter, linearised with
the Jacobian (checked numerically in the tests).

Association works on points, like TI's group tracker: each detection is gated
against each track with a Mahalanobis distance whose covariance includes the
track's estimated size and Doppler spread, and joins the closest one. Points
nobody claims are clustered with DBSCAN and start tentative tracks, confirmed
after 3 hits. Two rules keep one object to one track: unclaimed points close
to a confirmed track with a Doppler within a few m/s of it are its
micro-Doppler (limbs, wheels, rotors), and confirmed tracks that sit within a
vehicle length and move alike are merged.

**Micro-Doppler.** Parts of an object move differently from its body. A
pedestrian's legs swing from standstill to twice the walking speed, a bicycle
spoke moves at the wheel's rim speed, a rotor blade at 50+ m/s. Over time they
draw a signature around the body's Doppler line, a strong cue for
classification:

![micro-Doppler signatures](media/micro-doppler.png)

## 9. The laptop sonar

Sound in air is 343 m/s, so a 3.5 kHz sweep gives $\Delta R = c/2B$ = 4.9 cm,
not far from the 77 GHz radar. The sonar ([`sonar/processor.py`](../mira/sonar/processor.py))
reuses the same ideas with three twists:

**Periodic correlation.** The speaker loops one chirp of $N$ samples. Because
the transmitted signal is periodic, the microphone signal over *any* window of
$N$ samples contains each echo as a circular shift of the chirp. The circular
cross-correlation, computed with one FFT, gives the range profile without
knowing where the window starts relative to the transmission.

**Latency.** The sound card adds an unknown delay between playing and
recording. The direct speaker→microphone path is the first and strongest
peak, so ranges are measured from it, and the latency cancels.

**Range-Doppler coupling.** A hand moving at $v$ shifts the echo's frequency by
$f_d = -2vf_c/c$. With a chirp, a frequency offset is indistinguishable from a
time offset, $\Delta\tau = -f_d/S$, so the measured range is biased:

$$R_{measured} = R + v\,\frac{f_c T}{B}.$$

At 0.5 m/s that is 6 cm, larger than the resolution. A constant-velocity
Kalman filter on the measured range estimates $v$, and the bias is removed.
Radars have the same effect; at 77 GHz with a 25 µs chirp it is a few cm for a
car, which is why it is usually ignored there.

Moving objects are separated from the static room by a two-pulse canceller,
$|y_n - y_{n-1}|$, which leaves no ghost behind when the hand moves away. Its
price is *blind speeds*: when the hand moves exactly $\lambda/2$ between
chirps (0.41 m/s here) the phase wraps a full turn and the echo cancels for a
moment.

## 10. Further reading

- M. A. Richards, *Fundamentals of Radar Signal Processing*, McGraw-Hill. The reference for everything above.
- Texas Instruments, *The fundamentals of millimeter wave radar sensors* (SPYY005) and *MIMO radar* (SWRA554). Short and practical.
- H. Rohling, "Radar CFAR thresholding in clutter and multiple target situations", IEEE Trans. AES, 1983. Where OS-CFAR comes from.
- R. Schmidt, "Multiple emitter location and signal parameter estimation", IEEE Trans. AP, 1986 (MUSIC); T.-J. Shan, M. Wax, T. Kailath, "On spatial smoothing for direction-of-arrival estimation of coherent signals", IEEE Trans. ASSP, 1985.
- V. C. Chen, *The Micro-Doppler Effect in Radar*, Artech House.
- Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley.
