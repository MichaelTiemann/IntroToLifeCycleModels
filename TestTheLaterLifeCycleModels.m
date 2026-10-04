% Run to test all the later Life-Cycle Models are working without error.

if false

clear all
LifeCycleModel21

clear all
LifeCycleModel22

clear all
LifeCycleModel23

clear all
LifeCycleModel24

% There is no model 25 (yet)
% clear all
% LifeCycleModel25

clear all
LifeCycleModel26

clear all
LifeCycleModel27

clear all
LifeCycleModel28

clear all
LifeCycleModel29

clear all
LifeCycleModel30

%%
addpath('./Models31to35/')

clear all
LifeCycleModel31

clear all
LifeCycleModel32

clear all
LifeCycleModel33

clear all
LifeCycleModel34

clear all
LifeCycleModel35
end

%%
addpath('./Models36to39/')

clear all
LifeCycleModel36 % Fix this one

% clear all
% no GP preferences yet
% LifeCycleModel37

% clear all
% residualassets fails
% LifeCycleModel38 % Fix this one

% clear all
% no ambiguityaversion yet
% LifeCycleModel39

%%
addpath('./Models40to44/')

clear all
LifeCycleModel40

clear all
LifeCycleModel41

clear all
LifeCycleModel42

% clear all
% LifeCycleModel43

% clear all
% LifeCycleModel44

%% Test calibration -- these all take a long time
if false
addpath('./Models45to50/')

clear all
LifeCycleModel45

% clear all
% We don't have corr2cov from the financial toolbox
% LifeCycleModel46

% clear all
% We don't have PSIData
% LifeCycleModel47

clear all
LifeCycleModel48

clear all
LifeCycleModel49

clear all
LifeCycleModel50
end

%%
% Now the appendix models
addpath('./ModelsAppendixA/')

clear all
LifeCycleModelA1

clear all
LifeCycleModelA2

clear all
LifeCycleModelA3

clear all
LifeCycleModelA4

clear all
LifeCycleModelA5i

clear all
LifeCycleModelA5ii

clear all
LifeCycleModelA5iii

clear all
LifeCycleModelA5iv

clear all
LifeCycleModelA6

clear all
LifeCycleModelA7

clear all
LifeCycleModelA8

clear all
LifeCycleModelA9

disp("Skipping incomplete LifeCycleModelA10")
% The following requires `discretizeARpwGM_FarmerToda` which has yet to be
% written
% clear all
% LifeCycleModelA10

clear all
LifeCycleModelA11

clear all
LifeCycleModelA12

% Now the assignment models
addpath('./Assignments/')

clear all
Assignment1_LifeCycleModel

clear all
Assignment2_LifeCycleModel

% clear all
% Assignment3_LifeCycleModel

clear all
Assignment4_LifeCycleModel

