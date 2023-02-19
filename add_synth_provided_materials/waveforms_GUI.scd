(
/// some of the code isn't beautiful, but it might be helpful in starting to make sound and GUIs

//////// the first section defines the synth

///// the second section receives changes in the GUI settings and sets the synth sound

/// the third section is kind of overkill. its just setting up the window

s.boot;
//////////////////////////////////////////////////////////////////////
/////////////////////////////  SYNTH DEF  ////////////////////////////
//////////////////////////////////////////////////////////////////////

SynthDef(\oscillator, { |
	evenHarmonics = 1, oddHarmonics = 1, power = 1, activeHarmonics = 1,
	freq = 440 |
	var sig = Array.fill(25, {arg i;
		var multiple1;
		j = i + 1;
		if ((j % 2) == 1, {
			multiple1 = oddHarmonics;
		}, {
			multiple1 = evenHarmonics;
		});
		SinOsc.ar(
			freq: MouseX.kr(0.2, 15)*(freq*j),
			mul: MouseY.kr(0.01, 0.8)*(1/pow(j, power))*multiple1
		) * (j <= activeHarmonics);
	});
	Out.ar(0, sig*0.1 ! 2);
}).add;

//////////////////////////////////////////////////////////////////////
///////////////////////  SET SYNTH WITH GUI  /////////////////////////
//////////////////////////////////////////////////////////////////////

~setSynth = {arg frequency, oscillator, harmonics, onOff, onOffTouched;
	// oscillator: 0-sine 1-triangle 2-square 3-saw
	var thisFreq = frequency;

	(frequency + "  \t" + oscillator + "\t" + harmonics + "\t" + onOff).postln;

	if (onOff == 1, {
		// create a new synth only when first turned on
		if (onOffTouched == 1, { ~synth = Synth(\oscillator) });

		// if synth is on, update these settings when the function is called
		case
		{oscillator == 0}{
			~synth.set(\evenHarmonics, 0);
			~synth.set(\activeHarmonics, 1);
			~synth.set(\power, 1);
		}
		{oscillator == 1}{
			~synth.set(\evenHarmonics, 0);
			~synth.set(\activeHarmonics, harmonics);
			~synth.set(\power, 2);
		}
		{oscillator == 2}{
			~synth.set(\evenHarmonics, 0);
			~synth.set(\activeHarmonics, harmonics);
			~synth.set(\power, 1);
		}
		{oscillator == 3}{
			~synth.set(\evenHarmonics, 1);
			~synth.set(\activeHarmonics, harmonics);
			~synth.set(\power, 1);
		};

		~synth.set(\freq, thisFreq);

	}, { // free synth when onOff = 0
		if (onOffTouched == 1, {
			"freed".postln;
			~synth.free;
		});
	});
};


//////////////////////////////////////////////////////////////////////
/////////////// Everything below here is for the window /////////////////
//////////////////////////////////////////////////////////////////////

/////// its a complete mess... so yea... sorry //////

~clip = {arg min, max, input;
	var output;
	case
	{input < min}{output = min}
	{input > max}{output = max}
	{output = input};
	output;
};


~sizeWindow = {arg x_scale, y_scale, align;

	var screensize, x_align, y_align, x_start, x_size, y_start, y_size;

	x_align = align[0];
	y_align = align[1];
	if (align[0].isString, {
		case
		{x_align == "left"}{x_align = 0}
		{x_align == "center"}{x_align = 1}
		{x_align == "right"}{x_align = 2};
	});
	if (align[1].isString, {
		case
		{y_align == "bottom"}{y_align = 0}
		{y_align == "center"}{y_align = 1}
		{y_align == "top"}{y_align = 2};
	});

	screensize = [Window.screenBounds.width, Window.screenBounds.height];
	x_scale = ~clip.(0.05, 1, (x_scale/100));
	y_scale = ~clip.(0.05, 1, (y_scale/100));

	x_size = (x_scale*(screensize[0]));
	y_size = (y_scale*(screensize[1]));

	case
	{x_align == 0}{x_start = 0}
	{x_align == 1}{x_start = (screensize[0] - x_size)/2}
	{x_align == 2}{x_start = screensize[0] - x_size};

	case
	{y_align == 0}{y_start = 0}
	{y_align == 1}{y_start = (screensize[1] - y_size)/2}
	{y_align == 2}{y_start = screensize[1] - y_size};

	[x_start.round, y_start.round, x_size, y_size];
};


~makeWindow = {arg x_scale, y_scale, align;
	// x_scale, y_scale => integer 5...100 (percent of screen used)
	// align => array with two values: [ horizontal alignment, vertical alignment ]
	//   ex. [ 0 || 1 || 2, 0 || 1 || 2 ]  -or-  [ "left" || "center" || "right", "bottom" || "center" || "top" ]

	var window_data, resolution, x_pixels, y_pixels;
	var dict_TOP, dict_PANEL;
	var inc, dec, reset, startStop, toggleStartStop, oscMenu, numHarmonicsText,
	textFunc, setButtons;
	var height, width, x_pos, y_pos, exp;
	var activeHarmonics = 1;
	var frequency = 440;
	var plotter, freqscope, plotterResolution;
	window_data = ~sizeWindow.(x_scale, y_scale, align);
	(" window rect:"+window_data).postln;
	width = window_data[2];
	height = window_data[3];

	w = Window("pixelator", Rect(window_data[0], window_data[1], width, height)).front;
	w.view.background_(Color.new(0.5, 0.5, 0.5, 1));


	// Dictionaries of object positions
	// this was so I could move the panel around as one object without having to change each individual number
	// not the best way to do this... you can use a View inside a Window instead of keeping track of offset
	~panelOffsetX = 40;
	~panelOffsetY = 40;
	~buttonSize = 40;
	dict_PANEL= Dictionary.newFrom(List[
		\oscMenu, Rect(10+~panelOffsetX, 10+~panelOffsetY, 120, 20),
		\incButton, Rect(165+~panelOffsetX,10+~panelOffsetY,~buttonSize,~buttonSize),
		\decButton, Rect(~buttonSize+165+~panelOffsetX,10+~panelOffsetY,~buttonSize,~buttonSize),
		\resButton, Rect(265+~panelOffsetX,10+~panelOffsetY,70, ~buttonSize),
		\startStopButton, Rect(10+~panelOffsetX,45+~panelOffsetY,120,60),
		\numHarmonicsText, Rect(165+~panelOffsetX,70+~panelOffsetY,250,30)
	]);

	freqscope = FreqScopeView(w, Rect(0, height*0.34, width, height*0.6));
	freqscope.active_(true); // turn it on the first time
	w.onClose_({ freqscope.kill }); // you must have this


	/// GUI Panel ///

	oscMenu = PopUpMenu(w, dict_PANEL.at(\oscMenu));
	oscMenu.items = [
		"Sine", "Triangle", "Square", "Saw"
	];
	oscMenu.action = {
		~setSynth.(frequency, oscMenu.value, activeHarmonics, toggleStartStop, 0);
	};

	startStop = Button.new(w, dict_PANEL.at(\startStopButton)).font_(Font("Helvetica", 30, true)).string = "start";
	inc = Button.new(w, dict_PANEL.at(\incButton)).font_(Font("Helvetica", 24, true)).string = "+";
	dec = Button.new(w, dict_PANEL.at(\decButton)).font_(Font("Helvetica", 24, true)).string = "-";
	reset = Button.new(w, dict_PANEL.at(\resButton)).font_(Font("Helvetica", 15, true)).string = "reset";

	numHarmonicsText = StaticText.new(w, dict_PANEL.at(\numHarmonicsText));
	numHarmonicsText.string_("Active Harmonics:" + activeHarmonics);
	numHarmonicsText.font_(Font("Helvetica", 18, true));


	textFunc = { arg command;
		case
		{command == "inc"}{activeHarmonics = ~clip.(1, 25, activeHarmonics + 1);}
		{command == "dec"}{activeHarmonics = ~clip.(1, 25, activeHarmonics - 1);}
		{command == "reset"}{activeHarmonics = 1;};
		numHarmonicsText.string_("Active Harmonics:" + activeHarmonics);
		~setSynth.(frequency, oscMenu.value, activeHarmonics, toggleStartStop, 0);
	};

	inc.action = { textFunc.("inc") };
	dec.action = { textFunc.("dec") };
	reset.action = { textFunc.("reset") };

	// start and stop button
	toggleStartStop = 0;
	startStop.action = {
		if ( toggleStartStop == 0, {
			toggleStartStop = 1;
			startStop.string = "stop";
		},{
			toggleStartStop = 0;
			startStop.string = "start";
		});
		~setSynth.(frequency, oscMenu.value, activeHarmonics, toggleStartStop, 1);
	};

	w.drawFunc = {
		var x1 = 0+~panelOffsetX;
		var x2 = 140+~panelOffsetX;
		var x3 = 360+~panelOffsetX;
		var y1 = 0+~panelOffsetY;
		var y2 = 120+~panelOffsetY;

		//// border around the buttons in the GUI
		Pen.strokeColor = Color.new(0, 0.1, 0.3, 1);
		Pen.fillColor = Color.new(0.8, 0.8, 0.8, 0.3);
		Pen.width = 5;
		Pen.moveTo(x1@y1);
		Pen.lineTo(x1@y2);
		Pen.lineTo(x2@y2);
		Pen.lineTo(x2@y1);
		Pen.lineTo(x1@y1);
		Pen.moveTo(x2@y1);
		Pen.lineTo(x3@y1);
		Pen.lineTo(x3@y2);
		Pen.lineTo(x2@y2);
		Pen.fillStroke;
	};

	w.refresh;
};
// x size, y size, alignment
~makeWindow.(100, 100, ["left", "top"]);
)