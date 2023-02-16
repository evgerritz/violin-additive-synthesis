/*
This code shows server-side multiplication of an envelope (curve) by another (arbitrary) curve. This would be useful for modeling harmonic partial amplitudes over time where linear or exponential increases are not appropriate (real-sounding). Useful for FM or additive synthesis as well as experimental

*/

(
var curve = Env([0,1,0], [1,1], 'sine'); // the length of this doesn't matter -- only for shape.
var dcurve = curve.discretize; // convert to Signal which can be loaded to a Buffer.
var buf = Buffer.loadCollection(s, dcurve);
var bs = buf.numFrames;

{ // maybe a buffer? Cause can read forward and backward
	var bs = buf.numFrames;
	var env = Env([0, 1, 0], [1, 1]);
	var trigger = Impulse.ar(44100);
	var position = env.ar * BufFrames.ir(buf);
	var lcurve = BufRd.kr(1, buf, position);
	[env.kr, lcurve, env.kr * lcurve * 1.5];

}.plot(2)
)