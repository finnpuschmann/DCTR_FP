��
��
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.42v2.8.3-90-g1b8f5c396f08��
y
dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_namedense_0/kernel
r
"dense_0/kernel/Read/ReadVariableOpReadVariableOpdense_0/kernel*
_output_shapes
:	�d*
dtype0
p
dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_0/bias
i
 dense_0/bias/Read/ReadVariableOpReadVariableOpdense_0/bias*
_output_shapes
:d*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:dd*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:d*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:dd*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:d*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:d*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
x
tdist_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_nametdist_0/kernel
q
"tdist_0/kernel/Read/ReadVariableOpReadVariableOptdist_0/kernel*
_output_shapes

:d*
dtype0
p
tdist_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nametdist_0/bias
i
 tdist_0/bias/Read/ReadVariableOpReadVariableOptdist_0/bias*
_output_shapes
:d*
dtype0
x
tdist_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_nametdist_1/kernel
q
"tdist_1/kernel/Read/ReadVariableOpReadVariableOptdist_1/kernel*
_output_shapes

:dd*
dtype0
p
tdist_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nametdist_1/bias
i
 tdist_1/bias/Read/ReadVariableOpReadVariableOptdist_1/bias*
_output_shapes
:d*
dtype0
y
tdist_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*
shared_nametdist_2/kernel
r
"tdist_2/kernel/Read/ReadVariableOpReadVariableOptdist_2/kernel*
_output_shapes
:	d�*
dtype0
q
tdist_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametdist_2/bias
j
 tdist_2/bias/Read/ReadVariableOpReadVariableOptdist_2/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*&
shared_nameAdam/dense_0/kernel/m
�
)Adam/dense_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_0/kernel/m*
_output_shapes
:	�d*
dtype0
~
Adam/dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_0/bias/m
w
'Adam/dense_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_0/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:dd*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:dd*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:d*
dtype0
�
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:d*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
�
Adam/tdist_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/tdist_0/kernel/m

)Adam/tdist_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/tdist_0/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/tdist_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/tdist_0/bias/m
w
'Adam/tdist_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/tdist_0/bias/m*
_output_shapes
:d*
dtype0
�
Adam/tdist_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*&
shared_nameAdam/tdist_1/kernel/m

)Adam/tdist_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/tdist_1/kernel/m*
_output_shapes

:dd*
dtype0
~
Adam/tdist_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/tdist_1/bias/m
w
'Adam/tdist_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/tdist_1/bias/m*
_output_shapes
:d*
dtype0
�
Adam/tdist_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*&
shared_nameAdam/tdist_2/kernel/m
�
)Adam/tdist_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/tdist_2/kernel/m*
_output_shapes
:	d�*
dtype0

Adam/tdist_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/tdist_2/bias/m
x
'Adam/tdist_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/tdist_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*&
shared_nameAdam/dense_0/kernel/v
�
)Adam/dense_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_0/kernel/v*
_output_shapes
:	�d*
dtype0
~
Adam/dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_0/bias/v
w
'Adam/dense_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_0/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:dd*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:dd*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:d*
dtype0
�
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:d*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0
�
Adam/tdist_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/tdist_0/kernel/v

)Adam/tdist_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/tdist_0/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/tdist_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/tdist_0/bias/v
w
'Adam/tdist_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/tdist_0/bias/v*
_output_shapes
:d*
dtype0
�
Adam/tdist_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*&
shared_nameAdam/tdist_1/kernel/v

)Adam/tdist_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/tdist_1/kernel/v*
_output_shapes

:dd*
dtype0
~
Adam/tdist_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/tdist_1/bias/v
w
'Adam/tdist_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/tdist_1/bias/v*
_output_shapes
:d*
dtype0
�
Adam/tdist_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*&
shared_nameAdam/tdist_2/kernel/v
�
)Adam/tdist_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/tdist_2/kernel/v*
_output_shapes
:	d�*
dtype0

Adam/tdist_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/tdist_2/bias/v
x
'Adam/tdist_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/tdist_2/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
Є
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B�
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer_with_weights-6
layer-15
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	layer
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
�
	(layer
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
�
	5layer
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
�

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
�

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
�

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses*
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
�

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�beta_1
�beta_2

�decay
�learning_rate
	�iterNm�Om�\m�]m�jm�km�xm�ym�	�m�	�m�	�m�	�m�	�m�	�m�Nv�Ov�\v�]v�jv�kv�xv�yv�	�v�	�v�	�v�	�v�	�v�	�v�*
p
�0
�1
�2
�3
�4
�5
N6
O7
\8
]9
j10
k11
x12
y13*
p
�0
�1
�2
�3
�4
�5
N6
O7
\8
]9
j10
k11
x12
y13*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

�serving_default* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 
* 
* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 
* 
* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_0/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_0/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

N0
O1*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

\0
]1*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

j0
k1*

j0
k1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

x0
y1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEtdist_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEtdist_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEtdist_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEtdist_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEtdist_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEtdist_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*

�0
�1*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

(0*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

50*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

�total

�count
�	variables
�	keras_api*
M

�total

�count
�
_fn_kwargs
�	variables
�	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
�{
VARIABLE_VALUEAdam/dense_0/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_0/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/tdist_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/tdist_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/tdist_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/tdist_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/tdist_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/tdist_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_0/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_0/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/tdist_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/tdist_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/tdist_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/tdist_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/tdist_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/tdist_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_inputPlaceholder*4
_output_shapes"
 :������������������*
dtype0*)
shape :������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputtdist_0/kerneltdist_0/biastdist_1/kerneltdist_1/biastdist_2/kerneltdist_2/biasdense_0/kerneldense_0/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_569195
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_0/kernel/Read/ReadVariableOp dense_0/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp"tdist_0/kernel/Read/ReadVariableOp tdist_0/bias/Read/ReadVariableOp"tdist_1/kernel/Read/ReadVariableOp tdist_1/bias/Read/ReadVariableOp"tdist_2/kernel/Read/ReadVariableOp tdist_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_0/kernel/m/Read/ReadVariableOp'Adam/dense_0/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp)Adam/tdist_0/kernel/m/Read/ReadVariableOp'Adam/tdist_0/bias/m/Read/ReadVariableOp)Adam/tdist_1/kernel/m/Read/ReadVariableOp'Adam/tdist_1/bias/m/Read/ReadVariableOp)Adam/tdist_2/kernel/m/Read/ReadVariableOp'Adam/tdist_2/bias/m/Read/ReadVariableOp)Adam/dense_0/kernel/v/Read/ReadVariableOp'Adam/dense_0/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOp)Adam/tdist_0/kernel/v/Read/ReadVariableOp'Adam/tdist_0/bias/v/Read/ReadVariableOp)Adam/tdist_1/kernel/v/Read/ReadVariableOp'Adam/tdist_1/bias/v/Read/ReadVariableOp)Adam/tdist_2/kernel/v/Read/ReadVariableOp'Adam/tdist_2/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_569798
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_0/kerneldense_0/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasoutput/kerneloutput/biasbeta_1beta_2decaylearning_rate	Adam/itertdist_0/kerneltdist_0/biastdist_1/kerneltdist_1/biastdist_2/kerneltdist_2/biastotalcounttotal_1count_1Adam/dense_0/kernel/mAdam/dense_0/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/tdist_0/kernel/mAdam/tdist_0/bias/mAdam/tdist_1/kernel/mAdam/tdist_1/bias/mAdam/tdist_2/kernel/mAdam/tdist_2/bias/mAdam/dense_0/kernel/vAdam/dense_0/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/output/kernel/vAdam/output/bias/vAdam/tdist_0/kernel/vAdam/tdist_0/bias/vAdam/tdist_1/kernel/vAdam/tdist_1/bias/vAdam/tdist_2/kernel/vAdam/tdist_2/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_569961ƾ
�
�
C__inference_tdist_1_layer_call_and_return_conditional_losses_568191

inputs 
dense_1_568181:dd
dense_1_568183:d
identity��dense_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_568181dense_1_568183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568180\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������dh
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
�
C__inference_tdist_2_layer_call_and_return_conditional_losses_568272

inputs!
dense_2_568262:	d�
dense_2_568264:	�
identity��dense_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_2_568262dense_2_568264*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568261\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
�
C__inference_tdist_0_layer_call_and_return_conditional_losses_568110

inputs
dense_568100:d
dense_568102:d
identity��dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:����������
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_568100dense_568102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_568099\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������df
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�D
�
A__inference_model_layer_call_and_return_conditional_losses_568712

inputs 
tdist_0_568661:d
tdist_0_568663:d 
tdist_1_568669:dd
tdist_1_568671:d!
tdist_2_568677:	d�
tdist_2_568679:	�!
dense_0_568687:	�d
dense_0_568689:d 
dense_1_568693:dd
dense_1_568695:d 
dense_2_568699:dd
dense_2_568701:d
output_568705:d
output_568707:
identity��dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�tdist_0/StatefulPartitionedCall�tdist_1/StatefulPartitionedCall�tdist_2/StatefulPartitionedCall�
tdist_0/StatefulPartitionedCallStatefulPartitionedCallinputstdist_0_568661tdist_0_568663*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_0_layer_call_and_return_conditional_losses_568149f
tdist_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
tdist_0/ReshapeReshapeinputstdist_0/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
activation/PartitionedCallPartitionedCall(tdist_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_568337�
tdist_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0tdist_1_568669tdist_1_568671*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_1_layer_call_and_return_conditional_losses_568230f
tdist_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/ReshapeReshape#activation/PartitionedCall:output:0tdist_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
activation_1/PartitionedCallPartitionedCall(tdist_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_568351�
tdist_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0tdist_2_568677tdist_2_568679*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_2_layer_call_and_return_conditional_losses_568311f
tdist_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/ReshapeReshape%activation_1/PartitionedCall:output:0tdist_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
mask/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_mask_layer_call_and_return_conditional_losses_568608�
activation_2/PartitionedCallPartitionedCall(tdist_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_568376�
sum/PartitionedCallPartitionedCallmask/PartitionedCall:output:0%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_sum_layer_call_and_return_conditional_losses_568388�
dense_0/StatefulPartitionedCallStatefulPartitionedCallsum/PartitionedCall:output:0dense_0_568687dense_0_568689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_568400�
activation_3/PartitionedCallPartitionedCall(dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_568411�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_1_568693dense_1_568695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568423�
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_568434�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_2_568699dense_2_568701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568446�
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_568457�
output/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0output_568705output_568707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_568469�
activation_6/PartitionedCallPartitionedCall'output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_568480t
IdentityIdentity%activation_6/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^output/StatefulPartitionedCall ^tdist_0/StatefulPartitionedCall ^tdist_1/StatefulPartitionedCall ^tdist_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
tdist_0/StatefulPartitionedCalltdist_0/StatefulPartitionedCall2B
tdist_1/StatefulPartitionedCalltdist_1/StatefulPartitionedCall2B
tdist_2/StatefulPartitionedCalltdist_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_2_layer_call_and_return_conditional_losses_569433

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�������������������:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
I
-__inference_activation_5_layer_call_fn_569531

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_568457`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_569612

inputs
unknown:	d�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568261p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
d
H__inference_activation_3_layer_call_and_return_conditional_losses_568411

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������dZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
G
+__inference_activation_layer_call_fn_569260

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_568337m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������d:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
�
(__inference_tdist_0_layer_call_fn_569204

inputs
unknown:d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_0_layer_call_and_return_conditional_losses_568110|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
C__inference_tdist_1_layer_call_and_return_conditional_losses_569325

inputs8
&dense_1_matmul_readvariableop_resource:dd5
'dense_1_biasadd_readvariableop_resource:d
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
�
(__inference_tdist_2_layer_call_fn_569353

inputs
unknown:	d�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_2_layer_call_and_return_conditional_losses_568311}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_0_layer_call_and_return_conditional_losses_569468

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�D
�
A__inference_model_layer_call_and_return_conditional_losses_568830	
input 
tdist_0_568779:d
tdist_0_568781:d 
tdist_1_568787:dd
tdist_1_568789:d!
tdist_2_568795:	d�
tdist_2_568797:	�!
dense_0_568805:	�d
dense_0_568807:d 
dense_1_568811:dd
dense_1_568813:d 
dense_2_568817:dd
dense_2_568819:d
output_568823:d
output_568825:
identity��dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�tdist_0/StatefulPartitionedCall�tdist_1/StatefulPartitionedCall�tdist_2/StatefulPartitionedCall�
tdist_0/StatefulPartitionedCallStatefulPartitionedCallinputtdist_0_568779tdist_0_568781*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_0_layer_call_and_return_conditional_losses_568110f
tdist_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   s
tdist_0/ReshapeReshapeinputtdist_0/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
activation/PartitionedCallPartitionedCall(tdist_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_568337�
tdist_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0tdist_1_568787tdist_1_568789*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_1_layer_call_and_return_conditional_losses_568191f
tdist_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/ReshapeReshape#activation/PartitionedCall:output:0tdist_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
activation_1/PartitionedCallPartitionedCall(tdist_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_568351�
tdist_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0tdist_2_568795tdist_2_568797*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_2_layer_call_and_return_conditional_losses_568272f
tdist_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/ReshapeReshape%activation_1/PartitionedCall:output:0tdist_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
mask/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_mask_layer_call_and_return_conditional_losses_568369�
activation_2/PartitionedCallPartitionedCall(tdist_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_568376�
sum/PartitionedCallPartitionedCallmask/PartitionedCall:output:0%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_sum_layer_call_and_return_conditional_losses_568388�
dense_0/StatefulPartitionedCallStatefulPartitionedCallsum/PartitionedCall:output:0dense_0_568805dense_0_568807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_568400�
activation_3/PartitionedCallPartitionedCall(dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_568411�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_1_568811dense_1_568813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568423�
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_568434�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_2_568817dense_2_568819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568446�
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_568457�
output/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0output_568823output_568825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_568469�
activation_6/PartitionedCallPartitionedCall'output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_568480t
IdentityIdentity%activation_6/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^output/StatefulPartitionedCall ^tdist_0/StatefulPartitionedCall ^tdist_1/StatefulPartitionedCall ^tdist_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
tdist_0/StatefulPartitionedCalltdist_0/StatefulPartitionedCall2B
tdist_1/StatefulPartitionedCalltdist_1/StatefulPartitionedCall2B
tdist_2/StatefulPartitionedCalltdist_2/StatefulPartitionedCall:[ W
4
_output_shapes"
 :������������������

_user_specified_nameinput
�
A
%__inference_mask_layer_call_fn_569400

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_mask_layer_call_and_return_conditional_losses_568369i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
(__inference_tdist_2_layer_call_fn_569344

inputs
unknown:	d�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_2_layer_call_and_return_conditional_losses_568272}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
�
(__inference_dense_0_layer_call_fn_569458

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_568400o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_568099

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_568514	
input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:	d�
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:d

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_568483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
4
_output_shapes"
 :������������������

_user_specified_nameinput
�
A
%__inference_mask_layer_call_fn_569405

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_mask_layer_call_and_return_conditional_losses_568608i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
i
?__inference_sum_layer_call_and_return_conditional_losses_568388

inputs
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :x

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������m
MatMulBatchMatMulV2ExpandDims:output:0inputs_1*
T0*,
_output_shapes
:����������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:m
SqueezeSqueezeMatMul:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
Y
IdentityIdentitySqueeze:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:������������������:�������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
C__inference_tdist_2_layer_call_and_return_conditional_losses_568311

inputs!
dense_2_568301:	d�
dense_2_568303:	�
identity��dense_2/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_2_568301dense_2_568303*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568261\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������h
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_569526

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
d
H__inference_activation_5_layer_call_and_return_conditional_losses_569536

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������dZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_tdist_0_layer_call_and_return_conditional_losses_568149

inputs
dense_568139:d
dense_568141:d
identity��dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:����������
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_568139dense_568141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_568099\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������df
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_569961
file_prefix2
assignvariableop_dense_0_kernel:	�d-
assignvariableop_1_dense_0_bias:d3
!assignvariableop_2_dense_1_kernel:dd-
assignvariableop_3_dense_1_bias:d3
!assignvariableop_4_dense_2_kernel:dd-
assignvariableop_5_dense_2_bias:d2
 assignvariableop_6_output_kernel:d,
assignvariableop_7_output_bias:#
assignvariableop_8_beta_1: #
assignvariableop_9_beta_2: #
assignvariableop_10_decay: +
!assignvariableop_11_learning_rate: '
assignvariableop_12_adam_iter:	 4
"assignvariableop_13_tdist_0_kernel:d.
 assignvariableop_14_tdist_0_bias:d4
"assignvariableop_15_tdist_1_kernel:dd.
 assignvariableop_16_tdist_1_bias:d5
"assignvariableop_17_tdist_2_kernel:	d�/
 assignvariableop_18_tdist_2_bias:	�#
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: <
)assignvariableop_23_adam_dense_0_kernel_m:	�d5
'assignvariableop_24_adam_dense_0_bias_m:d;
)assignvariableop_25_adam_dense_1_kernel_m:dd5
'assignvariableop_26_adam_dense_1_bias_m:d;
)assignvariableop_27_adam_dense_2_kernel_m:dd5
'assignvariableop_28_adam_dense_2_bias_m:d:
(assignvariableop_29_adam_output_kernel_m:d4
&assignvariableop_30_adam_output_bias_m:;
)assignvariableop_31_adam_tdist_0_kernel_m:d5
'assignvariableop_32_adam_tdist_0_bias_m:d;
)assignvariableop_33_adam_tdist_1_kernel_m:dd5
'assignvariableop_34_adam_tdist_1_bias_m:d<
)assignvariableop_35_adam_tdist_2_kernel_m:	d�6
'assignvariableop_36_adam_tdist_2_bias_m:	�<
)assignvariableop_37_adam_dense_0_kernel_v:	�d5
'assignvariableop_38_adam_dense_0_bias_v:d;
)assignvariableop_39_adam_dense_1_kernel_v:dd5
'assignvariableop_40_adam_dense_1_bias_v:d;
)assignvariableop_41_adam_dense_2_kernel_v:dd5
'assignvariableop_42_adam_dense_2_bias_v:d:
(assignvariableop_43_adam_output_kernel_v:d4
&assignvariableop_44_adam_output_bias_v:;
)assignvariableop_45_adam_tdist_0_kernel_v:d5
'assignvariableop_46_adam_tdist_0_bias_v:d;
)assignvariableop_47_adam_tdist_1_kernel_v:dd5
'assignvariableop_48_adam_tdist_1_bias_v:d<
)assignvariableop_49_adam_tdist_2_kernel_v:	d�6
'assignvariableop_50_adam_tdist_2_bias_v:	�
identity_52��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_tdist_0_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp assignvariableop_14_tdist_0_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_tdist_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp assignvariableop_16_tdist_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_tdist_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp assignvariableop_18_tdist_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_0_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_0_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_output_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_output_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_tdist_0_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_tdist_0_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_tdist_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_tdist_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_tdist_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_tdist_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_0_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_0_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_output_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_output_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_tdist_0_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_tdist_0_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_tdist_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_tdist_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_tdist_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_tdist_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�{
�
!__inference__wrapped_model_568075	
inputD
2model_tdist_0_dense_matmul_readvariableop_resource:dA
3model_tdist_0_dense_biasadd_readvariableop_resource:dF
4model_tdist_1_dense_1_matmul_readvariableop_resource:ddC
5model_tdist_1_dense_1_biasadd_readvariableop_resource:dG
4model_tdist_2_dense_2_matmul_readvariableop_resource:	d�D
5model_tdist_2_dense_2_biasadd_readvariableop_resource:	�?
,model_dense_0_matmul_readvariableop_resource:	�d;
-model_dense_0_biasadd_readvariableop_resource:d>
,model_dense_1_matmul_readvariableop_resource:dd;
-model_dense_1_biasadd_readvariableop_resource:d>
,model_dense_2_matmul_readvariableop_resource:dd;
-model_dense_2_biasadd_readvariableop_resource:d=
+model_output_matmul_readvariableop_resource:d:
,model_output_biasadd_readvariableop_resource:
identity��$model/dense_0/BiasAdd/ReadVariableOp�#model/dense_0/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�#model/output/BiasAdd/ReadVariableOp�"model/output/MatMul/ReadVariableOp�*model/tdist_0/dense/BiasAdd/ReadVariableOp�)model/tdist_0/dense/MatMul/ReadVariableOp�,model/tdist_1/dense_1/BiasAdd/ReadVariableOp�+model/tdist_1/dense_1/MatMul/ReadVariableOp�,model/tdist_2/dense_2/BiasAdd/ReadVariableOp�+model/tdist_2/dense_2/MatMul/ReadVariableOpH
model/tdist_0/ShapeShapeinput*
T0*
_output_shapes
:k
!model/tdist_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#model/tdist_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/tdist_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/tdist_0/strided_sliceStridedSlicemodel/tdist_0/Shape:output:0*model/tdist_0/strided_slice/stack:output:0,model/tdist_0/strided_slice/stack_1:output:0,model/tdist_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
model/tdist_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   
model/tdist_0/ReshapeReshapeinput$model/tdist_0/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
)model/tdist_0/dense/MatMul/ReadVariableOpReadVariableOp2model_tdist_0_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
model/tdist_0/dense/MatMulMatMulmodel/tdist_0/Reshape:output:01model/tdist_0/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
*model/tdist_0/dense/BiasAdd/ReadVariableOpReadVariableOp3model_tdist_0_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/tdist_0/dense/BiasAddBiasAdd$model/tdist_0/dense/MatMul:product:02model/tdist_0/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dj
model/tdist_0/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������a
model/tdist_0/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
model/tdist_0/Reshape_1/shapePack(model/tdist_0/Reshape_1/shape/0:output:0$model/tdist_0/strided_slice:output:0(model/tdist_0/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/tdist_0/Reshape_1Reshape$model/tdist_0/dense/BiasAdd:output:0&model/tdist_0/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
model/tdist_0/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/tdist_0/Reshape_2Reshapeinput&model/tdist_0/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������~
model/activation/ReluRelu model/tdist_0/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������df
model/tdist_1/ShapeShape#model/activation/Relu:activations:0*
T0*
_output_shapes
:k
!model/tdist_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#model/tdist_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/tdist_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/tdist_1/strided_sliceStridedSlicemodel/tdist_1/Shape:output:0*model/tdist_1/strided_slice/stack:output:0,model/tdist_1/strided_slice/stack_1:output:0,model/tdist_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
model/tdist_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
model/tdist_1/ReshapeReshape#model/activation/Relu:activations:0$model/tdist_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
+model/tdist_1/dense_1/MatMul/ReadVariableOpReadVariableOp4model_tdist_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
model/tdist_1/dense_1/MatMulMatMulmodel/tdist_1/Reshape:output:03model/tdist_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
,model/tdist_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp5model_tdist_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/tdist_1/dense_1/BiasAddBiasAdd&model/tdist_1/dense_1/MatMul:product:04model/tdist_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dj
model/tdist_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������a
model/tdist_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
model/tdist_1/Reshape_1/shapePack(model/tdist_1/Reshape_1/shape/0:output:0$model/tdist_1/strided_slice:output:0(model/tdist_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/tdist_1/Reshape_1Reshape&model/tdist_1/dense_1/BiasAdd:output:0&model/tdist_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
model/tdist_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
model/tdist_1/Reshape_2Reshape#model/activation/Relu:activations:0&model/tdist_1/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������d�
model/activation_1/ReluRelu model/tdist_1/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������dh
model/tdist_2/ShapeShape%model/activation_1/Relu:activations:0*
T0*
_output_shapes
:k
!model/tdist_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#model/tdist_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/tdist_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/tdist_2/strided_sliceStridedSlicemodel/tdist_2/Shape:output:0*model/tdist_2/strided_slice/stack:output:0,model/tdist_2/strided_slice/stack_1:output:0,model/tdist_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
model/tdist_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
model/tdist_2/ReshapeReshape%model/activation_1/Relu:activations:0$model/tdist_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
+model/tdist_2/dense_2/MatMul/ReadVariableOpReadVariableOp4model_tdist_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
model/tdist_2/dense_2/MatMulMatMulmodel/tdist_2/Reshape:output:03model/tdist_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,model/tdist_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp5model_tdist_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/tdist_2/dense_2/BiasAddBiasAdd&model/tdist_2/dense_2/MatMul:product:04model/tdist_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������j
model/tdist_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������b
model/tdist_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
model/tdist_2/Reshape_1/shapePack(model/tdist_2/Reshape_1/shape/0:output:0$model/tdist_2/strided_slice:output:0(model/tdist_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/tdist_2/Reshape_1Reshape&model/tdist_2/dense_2/BiasAdd:output:0&model/tdist_2/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������n
model/tdist_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
model/tdist_2/Reshape_2Reshape%model/activation_1/Relu:activations:0&model/tdist_2/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������dZ
model/mask/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model/mask/NotEqualNotEqualinputmodel/mask/NotEqual/y:output:0*
T0*4
_output_shapes"
 :������������������k
 model/mask/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/mask/AnyAnymodel/mask/NotEqual:z:0)model/mask/Any/reduction_indices:output:0*0
_output_shapes
:������������������z
model/mask/CastCastmodel/mask/Any:output:0*

DstT0*

SrcT0
*0
_output_shapes
:�������������������
model/activation_2/ReluRelu model/tdist_2/Reshape_1:output:0*
T0*5
_output_shapes#
!:�������������������Z
model/sum/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/sum/ExpandDims
ExpandDimsmodel/mask/Cast:y:0!model/sum/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������
model/sum/MatMulBatchMatMulV2model/sum/ExpandDims:output:0%model/activation_2/Relu:activations:0*
T0*,
_output_shapes
:����������X
model/sum/ShapeShapemodel/sum/MatMul:output:0*
T0*
_output_shapes
:�
model/sum/SqueezeSqueezemodel/sum/MatMul:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
�
#model/dense_0/MatMul/ReadVariableOpReadVariableOp,model_dense_0_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
model/dense_0/MatMulMatMulmodel/sum/Squeeze:output:0+model/dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
$model/dense_0/BiasAdd/ReadVariableOpReadVariableOp-model_dense_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/dense_0/BiasAddBiasAddmodel/dense_0/MatMul:product:0,model/dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dq
model/activation_3/ReluRelumodel/dense_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
model/dense_1/MatMulMatMul%model/activation_3/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dq
model/activation_4/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
model/dense_2/MatMulMatMul%model/activation_4/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dq
model/activation_5/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
model/output/MatMulMatMul%model/activation_5/Relu:activations:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
model/activation_6/SoftmaxSoftmaxmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$model/activation_6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^model/dense_0/BiasAdd/ReadVariableOp$^model/dense_0/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp+^model/tdist_0/dense/BiasAdd/ReadVariableOp*^model/tdist_0/dense/MatMul/ReadVariableOp-^model/tdist_1/dense_1/BiasAdd/ReadVariableOp,^model/tdist_1/dense_1/MatMul/ReadVariableOp-^model/tdist_2/dense_2/BiasAdd/ReadVariableOp,^model/tdist_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 2L
$model/dense_0/BiasAdd/ReadVariableOp$model/dense_0/BiasAdd/ReadVariableOp2J
#model/dense_0/MatMul/ReadVariableOp#model/dense_0/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp2X
*model/tdist_0/dense/BiasAdd/ReadVariableOp*model/tdist_0/dense/BiasAdd/ReadVariableOp2V
)model/tdist_0/dense/MatMul/ReadVariableOp)model/tdist_0/dense/MatMul/ReadVariableOp2\
,model/tdist_1/dense_1/BiasAdd/ReadVariableOp,model/tdist_1/dense_1/BiasAdd/ReadVariableOp2Z
+model/tdist_1/dense_1/MatMul/ReadVariableOp+model/tdist_1/dense_1/MatMul/ReadVariableOp2\
,model/tdist_2/dense_2/BiasAdd/ReadVariableOp,model/tdist_2/dense_2/BiasAdd/ReadVariableOp2Z
+model/tdist_2/dense_2/MatMul/ReadVariableOp+model/tdist_2/dense_2/MatMul/ReadVariableOp:[ W
4
_output_shapes"
 :������������������

_user_specified_nameinput
�
�
C__inference_tdist_1_layer_call_and_return_conditional_losses_568230

inputs 
dense_1_568220:dd
dense_1_568222:d
identity��dense_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_568220dense_1_568222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568180\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������dh
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_568423

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_568261

inputs1
matmul_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_569487

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
B__inference_output_layer_call_and_return_conditional_losses_569555

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_568180

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�p
�
A__inference_model_layer_call_and_return_conditional_losses_569058

inputs>
,tdist_0_dense_matmul_readvariableop_resource:d;
-tdist_0_dense_biasadd_readvariableop_resource:d@
.tdist_1_dense_1_matmul_readvariableop_resource:dd=
/tdist_1_dense_1_biasadd_readvariableop_resource:dA
.tdist_2_dense_2_matmul_readvariableop_resource:	d�>
/tdist_2_dense_2_biasadd_readvariableop_resource:	�9
&dense_0_matmul_readvariableop_resource:	�d5
'dense_0_biasadd_readvariableop_resource:d8
&dense_1_matmul_readvariableop_resource:dd5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:dd5
'dense_2_biasadd_readvariableop_resource:d7
%output_matmul_readvariableop_resource:d4
&output_biasadd_readvariableop_resource:
identity��dense_0/BiasAdd/ReadVariableOp�dense_0/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�$tdist_0/dense/BiasAdd/ReadVariableOp�#tdist_0/dense/MatMul/ReadVariableOp�&tdist_1/dense_1/BiasAdd/ReadVariableOp�%tdist_1/dense_1/MatMul/ReadVariableOp�&tdist_2/dense_2/BiasAdd/ReadVariableOp�%tdist_2/dense_2/MatMul/ReadVariableOpC
tdist_0/ShapeShapeinputs*
T0*
_output_shapes
:e
tdist_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:g
tdist_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tdist_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
tdist_0/strided_sliceStridedSlicetdist_0/Shape:output:0$tdist_0/strided_slice/stack:output:0&tdist_0/strided_slice/stack_1:output:0&tdist_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
tdist_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
tdist_0/ReshapeReshapeinputstdist_0/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
#tdist_0/dense/MatMul/ReadVariableOpReadVariableOp,tdist_0_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
tdist_0/dense/MatMulMatMultdist_0/Reshape:output:0+tdist_0/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
$tdist_0/dense/BiasAdd/ReadVariableOpReadVariableOp-tdist_0_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
tdist_0/dense/BiasAddBiasAddtdist_0/dense/MatMul:product:0,tdist_0/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
tdist_0/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������[
tdist_0/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
tdist_0/Reshape_1/shapePack"tdist_0/Reshape_1/shape/0:output:0tdist_0/strided_slice:output:0"tdist_0/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
tdist_0/Reshape_1Reshapetdist_0/dense/BiasAdd:output:0 tdist_0/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dh
tdist_0/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   x
tdist_0/Reshape_2Reshapeinputs tdist_0/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������r
activation/ReluRelutdist_0/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������dZ
tdist_1/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:e
tdist_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:g
tdist_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tdist_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
tdist_1/strided_sliceStridedSlicetdist_1/Shape:output:0$tdist_1/strided_slice/stack:output:0&tdist_1/strided_slice/stack_1:output:0&tdist_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
tdist_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/ReshapeReshapeactivation/Relu:activations:0tdist_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
%tdist_1/dense_1/MatMul/ReadVariableOpReadVariableOp.tdist_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
tdist_1/dense_1/MatMulMatMultdist_1/Reshape:output:0-tdist_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
&tdist_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/tdist_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
tdist_1/dense_1/BiasAddBiasAdd tdist_1/dense_1/MatMul:product:0.tdist_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
tdist_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������[
tdist_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
tdist_1/Reshape_1/shapePack"tdist_1/Reshape_1/shape/0:output:0tdist_1/strided_slice:output:0"tdist_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
tdist_1/Reshape_1Reshape tdist_1/dense_1/BiasAdd:output:0 tdist_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dh
tdist_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/Reshape_2Reshapeactivation/Relu:activations:0 tdist_1/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������dt
activation_1/ReluRelutdist_1/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������d\
tdist_2/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:e
tdist_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:g
tdist_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tdist_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
tdist_2/strided_sliceStridedSlicetdist_2/Shape:output:0$tdist_2/strided_slice/stack:output:0&tdist_2/strided_slice/stack_1:output:0&tdist_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
tdist_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/ReshapeReshapeactivation_1/Relu:activations:0tdist_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
%tdist_2/dense_2/MatMul/ReadVariableOpReadVariableOp.tdist_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
tdist_2/dense_2/MatMulMatMultdist_2/Reshape:output:0-tdist_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&tdist_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/tdist_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
tdist_2/dense_2/BiasAddBiasAdd tdist_2/dense_2/MatMul:product:0.tdist_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
tdist_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������\
tdist_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
tdist_2/Reshape_1/shapePack"tdist_2/Reshape_1/shape/0:output:0tdist_2/strided_slice:output:0"tdist_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
tdist_2/Reshape_1Reshape tdist_2/dense_2/BiasAdd:output:0 tdist_2/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
tdist_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/Reshape_2Reshapeactivation_1/Relu:activations:0 tdist_2/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������dT
mask/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    z
mask/NotEqualNotEqualinputsmask/NotEqual/y:output:0*
T0*4
_output_shapes"
 :������������������e
mask/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������y
mask/AnyAnymask/NotEqual:z:0#mask/Any/reduction_indices:output:0*0
_output_shapes
:������������������n
	mask/CastCastmask/Any:output:0*

DstT0*

SrcT0
*0
_output_shapes
:������������������u
activation_2/ReluRelutdist_2/Reshape_1:output:0*
T0*5
_output_shapes#
!:�������������������T
sum/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
sum/ExpandDims
ExpandDimsmask/Cast:y:0sum/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������

sum/MatMulBatchMatMulV2sum/ExpandDims:output:0activation_2/Relu:activations:0*
T0*,
_output_shapes
:����������L
	sum/ShapeShapesum/MatMul:output:0*
T0*
_output_shapes
:u
sum/SqueezeSqueezesum/MatMul:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
dense_0/MatMulMatMulsum/Squeeze:output:0%dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������de
activation_3/ReluReludense_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_1/MatMulMatMulactivation_3/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������de
activation_4/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_2/MatMulMatMulactivation_4/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������de
activation_5/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
output/MatMulMatMulactivation_5/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
activation_6/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������m
IdentityIdentityactivation_6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp%^tdist_0/dense/BiasAdd/ReadVariableOp$^tdist_0/dense/MatMul/ReadVariableOp'^tdist_1/dense_1/BiasAdd/ReadVariableOp&^tdist_1/dense_1/MatMul/ReadVariableOp'^tdist_2/dense_2/BiasAdd/ReadVariableOp&^tdist_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2L
$tdist_0/dense/BiasAdd/ReadVariableOp$tdist_0/dense/BiasAdd/ReadVariableOp2J
#tdist_0/dense/MatMul/ReadVariableOp#tdist_0/dense/MatMul/ReadVariableOp2P
&tdist_1/dense_1/BiasAdd/ReadVariableOp&tdist_1/dense_1/BiasAdd/ReadVariableOp2N
%tdist_1/dense_1/MatMul/ReadVariableOp%tdist_1/dense_1/MatMul/ReadVariableOp2P
&tdist_2/dense_2/BiasAdd/ReadVariableOp&tdist_2/dense_2/BiasAdd/ReadVariableOp2N
%tdist_2/dense_2/MatMul/ReadVariableOp%tdist_2/dense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_568446

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
(__inference_tdist_0_layer_call_fn_569213

inputs
unknown:d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_0_layer_call_and_return_conditional_losses_568149|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
C__inference_tdist_0_layer_call_and_return_conditional_losses_569234

inputs6
$dense_matmul_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:d
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
\
@__inference_mask_layer_call_and_return_conditional_losses_569414

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    p
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*4
_output_shapes"
 :������������������`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*0
_output_shapes
:������������������d
CastCastAny:output:0*

DstT0*

SrcT0
*0
_output_shapes
:������������������Y
IdentityIdentityCast:y:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_1_layer_call_and_return_conditional_losses_568351

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :������������������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������d:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
\
@__inference_mask_layer_call_and_return_conditional_losses_568608

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    p
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*4
_output_shapes"
 :������������������`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*0
_output_shapes
:������������������d
CastCastAny:output:0*

DstT0*

SrcT0
*0
_output_shapes
:������������������Y
IdentityIdentityCast:y:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�D
�
A__inference_model_layer_call_and_return_conditional_losses_568884	
input 
tdist_0_568833:d
tdist_0_568835:d 
tdist_1_568841:dd
tdist_1_568843:d!
tdist_2_568849:	d�
tdist_2_568851:	�!
dense_0_568859:	�d
dense_0_568861:d 
dense_1_568865:dd
dense_1_568867:d 
dense_2_568871:dd
dense_2_568873:d
output_568877:d
output_568879:
identity��dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�tdist_0/StatefulPartitionedCall�tdist_1/StatefulPartitionedCall�tdist_2/StatefulPartitionedCall�
tdist_0/StatefulPartitionedCallStatefulPartitionedCallinputtdist_0_568833tdist_0_568835*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_0_layer_call_and_return_conditional_losses_568149f
tdist_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   s
tdist_0/ReshapeReshapeinputtdist_0/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
activation/PartitionedCallPartitionedCall(tdist_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_568337�
tdist_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0tdist_1_568841tdist_1_568843*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_1_layer_call_and_return_conditional_losses_568230f
tdist_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/ReshapeReshape#activation/PartitionedCall:output:0tdist_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
activation_1/PartitionedCallPartitionedCall(tdist_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_568351�
tdist_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0tdist_2_568849tdist_2_568851*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_2_layer_call_and_return_conditional_losses_568311f
tdist_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/ReshapeReshape%activation_1/PartitionedCall:output:0tdist_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
mask/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_mask_layer_call_and_return_conditional_losses_568608�
activation_2/PartitionedCallPartitionedCall(tdist_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_568376�
sum/PartitionedCallPartitionedCallmask/PartitionedCall:output:0%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_sum_layer_call_and_return_conditional_losses_568388�
dense_0/StatefulPartitionedCallStatefulPartitionedCallsum/PartitionedCall:output:0dense_0_568859dense_0_568861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_568400�
activation_3/PartitionedCallPartitionedCall(dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_568411�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_1_568865dense_1_568867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568423�
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_568434�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_2_568871dense_2_568873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568446�
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_568457�
output/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0output_568877output_568879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_568469�
activation_6/PartitionedCallPartitionedCall'output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_568480t
IdentityIdentity%activation_6/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^output/StatefulPartitionedCall ^tdist_0/StatefulPartitionedCall ^tdist_1/StatefulPartitionedCall ^tdist_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
tdist_0/StatefulPartitionedCalltdist_0/StatefulPartitionedCall2B
tdist_1/StatefulPartitionedCalltdist_1/StatefulPartitionedCall2B
tdist_2/StatefulPartitionedCalltdist_2/StatefulPartitionedCall:[ W
4
_output_shapes"
 :������������������

_user_specified_nameinput
�
�
(__inference_tdist_1_layer_call_fn_569274

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_1_layer_call_and_return_conditional_losses_568191|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
d
H__inference_activation_6_layer_call_and_return_conditional_losses_569565

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_activation_2_layer_call_and_return_conditional_losses_568376

inputs
identityT
ReluReluinputs*
T0*5
_output_shapes#
!:�������������������h
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�������������������:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
b
F__inference_activation_layer_call_and_return_conditional_losses_569265

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :������������������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������d:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_0_layer_call_and_return_conditional_losses_568400

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_tdist_1_layer_call_and_return_conditional_losses_569304

inputs8
&dense_1_matmul_readvariableop_resource:dd5
'dense_1_biasadd_readvariableop_resource:d
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapedense_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_569593

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_568923

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:	d�
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:d

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_568483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_568956

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:	d�
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:d

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_568712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_569574

inputs
unknown:d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_568099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_activation_3_layer_call_and_return_conditional_losses_569478

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������dZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_tdist_2_layer_call_and_return_conditional_losses_569395

inputs9
&dense_2_matmul_readvariableop_resource:	d�6
'dense_2_biasadd_readvariableop_resource:	�
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_569497

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
I
-__inference_activation_3_layer_call_fn_569473

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_568411`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
d
H__inference_activation_6_layer_call_and_return_conditional_losses_568480

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_activation_layer_call_and_return_conditional_losses_568337

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :������������������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������d:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
d
H__inference_activation_1_layer_call_and_return_conditional_losses_569335

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :������������������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������d:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
I
-__inference_activation_2_layer_call_fn_569428

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_568376n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�������������������:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_569584

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_569516

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568446o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_568776	
input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:	d�
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:d

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_568712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
4
_output_shapes"
 :������������������

_user_specified_nameinput
�D
�
A__inference_model_layer_call_and_return_conditional_losses_568483

inputs 
tdist_0_568325:d
tdist_0_568327:d 
tdist_1_568339:dd
tdist_1_568341:d!
tdist_2_568353:	d�
tdist_2_568355:	�!
dense_0_568401:	�d
dense_0_568403:d 
dense_1_568424:dd
dense_1_568426:d 
dense_2_568447:dd
dense_2_568449:d
output_568470:d
output_568472:
identity��dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�tdist_0/StatefulPartitionedCall�tdist_1/StatefulPartitionedCall�tdist_2/StatefulPartitionedCall�
tdist_0/StatefulPartitionedCallStatefulPartitionedCallinputstdist_0_568325tdist_0_568327*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_0_layer_call_and_return_conditional_losses_568110f
tdist_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
tdist_0/ReshapeReshapeinputstdist_0/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
activation/PartitionedCallPartitionedCall(tdist_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_568337�
tdist_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0tdist_1_568339tdist_1_568341*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_1_layer_call_and_return_conditional_losses_568191f
tdist_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/ReshapeReshape#activation/PartitionedCall:output:0tdist_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
activation_1/PartitionedCallPartitionedCall(tdist_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_568351�
tdist_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0tdist_2_568353tdist_2_568355*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_2_layer_call_and_return_conditional_losses_568272f
tdist_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/ReshapeReshape%activation_1/PartitionedCall:output:0tdist_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
mask/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_mask_layer_call_and_return_conditional_losses_568369�
activation_2/PartitionedCallPartitionedCall(tdist_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_568376�
sum/PartitionedCallPartitionedCallmask/PartitionedCall:output:0%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_sum_layer_call_and_return_conditional_losses_568388�
dense_0/StatefulPartitionedCallStatefulPartitionedCallsum/PartitionedCall:output:0dense_0_568401dense_0_568403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_0_layer_call_and_return_conditional_losses_568400�
activation_3/PartitionedCallPartitionedCall(dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_568411�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0dense_1_568424dense_1_568426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_568423�
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_568434�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dense_2_568447dense_2_568449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_568446�
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_568457�
output/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0output_568470output_568472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_568469�
activation_6/PartitionedCallPartitionedCall'output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_568480t
IdentityIdentity%activation_6/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^output/StatefulPartitionedCall ^tdist_0/StatefulPartitionedCall ^tdist_1/StatefulPartitionedCall ^tdist_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
tdist_0/StatefulPartitionedCalltdist_0/StatefulPartitionedCall2B
tdist_1/StatefulPartitionedCalltdist_1/StatefulPartitionedCall2B
tdist_2/StatefulPartitionedCalltdist_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_5_layer_call_and_return_conditional_losses_568457

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������dZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
(__inference_tdist_1_layer_call_fn_569283

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_tdist_1_layer_call_and_return_conditional_losses_568230|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�
k
?__inference_sum_layer_call_and_return_conditional_losses_569449
inputs_0
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :z

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������m
MatMulBatchMatMulV2ExpandDims:output:0inputs_1*
T0*,
_output_shapes
:����������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:m
SqueezeSqueezeMatMul:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
Y
IdentityIdentitySqueeze:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:������������������:�������������������:Z V
0
_output_shapes
:������������������
"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/1
�
�
C__inference_tdist_2_layer_call_and_return_conditional_losses_569374

inputs9
&dense_2_matmul_readvariableop_resource:	d�6
'dense_2_biasadd_readvariableop_resource:	�
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������d�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������d: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_569622

inputs1
matmul_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�p
�
A__inference_model_layer_call_and_return_conditional_losses_569160

inputs>
,tdist_0_dense_matmul_readvariableop_resource:d;
-tdist_0_dense_biasadd_readvariableop_resource:d@
.tdist_1_dense_1_matmul_readvariableop_resource:dd=
/tdist_1_dense_1_biasadd_readvariableop_resource:dA
.tdist_2_dense_2_matmul_readvariableop_resource:	d�>
/tdist_2_dense_2_biasadd_readvariableop_resource:	�9
&dense_0_matmul_readvariableop_resource:	�d5
'dense_0_biasadd_readvariableop_resource:d8
&dense_1_matmul_readvariableop_resource:dd5
'dense_1_biasadd_readvariableop_resource:d8
&dense_2_matmul_readvariableop_resource:dd5
'dense_2_biasadd_readvariableop_resource:d7
%output_matmul_readvariableop_resource:d4
&output_biasadd_readvariableop_resource:
identity��dense_0/BiasAdd/ReadVariableOp�dense_0/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�$tdist_0/dense/BiasAdd/ReadVariableOp�#tdist_0/dense/MatMul/ReadVariableOp�&tdist_1/dense_1/BiasAdd/ReadVariableOp�%tdist_1/dense_1/MatMul/ReadVariableOp�&tdist_2/dense_2/BiasAdd/ReadVariableOp�%tdist_2/dense_2/MatMul/ReadVariableOpC
tdist_0/ShapeShapeinputs*
T0*
_output_shapes
:e
tdist_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:g
tdist_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tdist_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
tdist_0/strided_sliceStridedSlicetdist_0/Shape:output:0$tdist_0/strided_slice/stack:output:0&tdist_0/strided_slice/stack_1:output:0&tdist_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
tdist_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
tdist_0/ReshapeReshapeinputstdist_0/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
#tdist_0/dense/MatMul/ReadVariableOpReadVariableOp,tdist_0_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
tdist_0/dense/MatMulMatMultdist_0/Reshape:output:0+tdist_0/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
$tdist_0/dense/BiasAdd/ReadVariableOpReadVariableOp-tdist_0_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
tdist_0/dense/BiasAddBiasAddtdist_0/dense/MatMul:product:0,tdist_0/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
tdist_0/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������[
tdist_0/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
tdist_0/Reshape_1/shapePack"tdist_0/Reshape_1/shape/0:output:0tdist_0/strided_slice:output:0"tdist_0/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
tdist_0/Reshape_1Reshapetdist_0/dense/BiasAdd:output:0 tdist_0/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dh
tdist_0/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   x
tdist_0/Reshape_2Reshapeinputs tdist_0/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������r
activation/ReluRelutdist_0/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������dZ
tdist_1/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:e
tdist_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:g
tdist_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tdist_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
tdist_1/strided_sliceStridedSlicetdist_1/Shape:output:0$tdist_1/strided_slice/stack:output:0&tdist_1/strided_slice/stack_1:output:0&tdist_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
tdist_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/ReshapeReshapeactivation/Relu:activations:0tdist_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
%tdist_1/dense_1/MatMul/ReadVariableOpReadVariableOp.tdist_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
tdist_1/dense_1/MatMulMatMultdist_1/Reshape:output:0-tdist_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
&tdist_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/tdist_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
tdist_1/dense_1/BiasAddBiasAdd tdist_1/dense_1/MatMul:product:0.tdist_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
tdist_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������[
tdist_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
tdist_1/Reshape_1/shapePack"tdist_1/Reshape_1/shape/0:output:0tdist_1/strided_slice:output:0"tdist_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
tdist_1/Reshape_1Reshape tdist_1/dense_1/BiasAdd:output:0 tdist_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dh
tdist_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_1/Reshape_2Reshapeactivation/Relu:activations:0 tdist_1/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������dt
activation_1/ReluRelutdist_1/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������d\
tdist_2/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:e
tdist_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:g
tdist_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tdist_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
tdist_2/strided_sliceStridedSlicetdist_2/Shape:output:0$tdist_2/strided_slice/stack:output:0&tdist_2/strided_slice/stack_1:output:0&tdist_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
tdist_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/ReshapeReshapeactivation_1/Relu:activations:0tdist_2/Reshape/shape:output:0*
T0*'
_output_shapes
:���������d�
%tdist_2/dense_2/MatMul/ReadVariableOpReadVariableOp.tdist_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
tdist_2/dense_2/MatMulMatMultdist_2/Reshape:output:0-tdist_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&tdist_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/tdist_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
tdist_2/dense_2/BiasAddBiasAdd tdist_2/dense_2/MatMul:product:0.tdist_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
tdist_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������\
tdist_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
tdist_2/Reshape_1/shapePack"tdist_2/Reshape_1/shape/0:output:0tdist_2/strided_slice:output:0"tdist_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
tdist_2/Reshape_1Reshape tdist_2/dense_2/BiasAdd:output:0 tdist_2/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:�������������������h
tdist_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
tdist_2/Reshape_2Reshapeactivation_1/Relu:activations:0 tdist_2/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������dT
mask/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    z
mask/NotEqualNotEqualinputsmask/NotEqual/y:output:0*
T0*4
_output_shapes"
 :������������������e
mask/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������y
mask/AnyAnymask/NotEqual:z:0#mask/Any/reduction_indices:output:0*0
_output_shapes
:������������������n
	mask/CastCastmask/Any:output:0*

DstT0*

SrcT0
*0
_output_shapes
:������������������u
activation_2/ReluRelutdist_2/Reshape_1:output:0*
T0*5
_output_shapes#
!:�������������������T
sum/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
sum/ExpandDims
ExpandDimsmask/Cast:y:0sum/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������

sum/MatMulBatchMatMulV2sum/ExpandDims:output:0activation_2/Relu:activations:0*
T0*,
_output_shapes
:����������L
	sum/ShapeShapesum/MatMul:output:0*
T0*
_output_shapes
:u
sum/SqueezeSqueezesum/MatMul:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
dense_0/MatMulMatMulsum/Squeeze:output:0%dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������de
activation_3/ReluReludense_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_1/MatMulMatMulactivation_3/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������de
activation_4/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0�
dense_2/MatMulMatMulactivation_4/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������de
activation_5/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
output/MatMulMatMulactivation_5/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
activation_6/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������m
IdentityIdentityactivation_6/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp%^tdist_0/dense/BiasAdd/ReadVariableOp$^tdist_0/dense/MatMul/ReadVariableOp'^tdist_1/dense_1/BiasAdd/ReadVariableOp&^tdist_1/dense_1/MatMul/ReadVariableOp'^tdist_2/dense_2/BiasAdd/ReadVariableOp&^tdist_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2L
$tdist_0/dense/BiasAdd/ReadVariableOp$tdist_0/dense/BiasAdd/ReadVariableOp2J
#tdist_0/dense/MatMul/ReadVariableOp#tdist_0/dense/MatMul/ReadVariableOp2P
&tdist_1/dense_1/BiasAdd/ReadVariableOp&tdist_1/dense_1/BiasAdd/ReadVariableOp2N
%tdist_1/dense_1/MatMul/ReadVariableOp%tdist_1/dense_1/MatMul/ReadVariableOp2P
&tdist_2/dense_2/BiasAdd/ReadVariableOp&tdist_2/dense_2/BiasAdd/ReadVariableOp2N
%tdist_2/dense_2/MatMul/ReadVariableOp%tdist_2/dense_2/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
I
-__inference_activation_6_layer_call_fn_569560

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_568480`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
\
@__inference_mask_layer_call_and_return_conditional_losses_568369

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    p
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*4
_output_shapes"
 :������������������`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*0
_output_shapes
:������������������d
CastCastAny:output:0*

DstT0*

SrcT0
*0
_output_shapes
:������������������Y
IdentityIdentityCast:y:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_4_layer_call_and_return_conditional_losses_569507

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������dZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_569195	
input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:	d�
	unknown_4:	�
	unknown_5:	�d
	unknown_6:d
	unknown_7:dd
	unknown_8:d
	unknown_9:dd

unknown_10:d

unknown_11:d

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_568075o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
4
_output_shapes"
 :������������������

_user_specified_nameinput
�
�
C__inference_tdist_0_layer_call_and_return_conditional_losses_569255

inputs6
$dense_matmul_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:d
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
���������S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d�
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :������������������dn
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :������������������d�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
I
-__inference_activation_1_layer_call_fn_569330

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_568351m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������d:\ X
4
_output_shapes"
 :������������������d
 
_user_specified_nameinputs
�a
�
__inference__traced_save_569798
file_prefix-
)savev2_dense_0_kernel_read_readvariableop+
'savev2_dense_0_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	-
)savev2_tdist_0_kernel_read_readvariableop+
'savev2_tdist_0_bias_read_readvariableop-
)savev2_tdist_1_kernel_read_readvariableop+
'savev2_tdist_1_bias_read_readvariableop-
)savev2_tdist_2_kernel_read_readvariableop+
'savev2_tdist_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_0_kernel_m_read_readvariableop2
.savev2_adam_dense_0_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop4
0savev2_adam_tdist_0_kernel_m_read_readvariableop2
.savev2_adam_tdist_0_bias_m_read_readvariableop4
0savev2_adam_tdist_1_kernel_m_read_readvariableop2
.savev2_adam_tdist_1_bias_m_read_readvariableop4
0savev2_adam_tdist_2_kernel_m_read_readvariableop2
.savev2_adam_tdist_2_bias_m_read_readvariableop4
0savev2_adam_dense_0_kernel_v_read_readvariableop2
.savev2_adam_dense_0_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop4
0savev2_adam_tdist_0_kernel_v_read_readvariableop2
.savev2_adam_tdist_0_bias_v_read_readvariableop4
0savev2_adam_tdist_1_kernel_v_read_readvariableop2
.savev2_adam_tdist_1_bias_v_read_readvariableop4
0savev2_adam_tdist_2_kernel_v_read_readvariableop2
.savev2_adam_tdist_2_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_0_kernel_read_readvariableop'savev2_dense_0_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop)savev2_tdist_0_kernel_read_readvariableop'savev2_tdist_0_bias_read_readvariableop)savev2_tdist_1_kernel_read_readvariableop'savev2_tdist_1_bias_read_readvariableop)savev2_tdist_2_kernel_read_readvariableop'savev2_tdist_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_0_kernel_m_read_readvariableop.savev2_adam_dense_0_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop0savev2_adam_tdist_0_kernel_m_read_readvariableop.savev2_adam_tdist_0_bias_m_read_readvariableop0savev2_adam_tdist_1_kernel_m_read_readvariableop.savev2_adam_tdist_1_bias_m_read_readvariableop0savev2_adam_tdist_2_kernel_m_read_readvariableop.savev2_adam_tdist_2_bias_m_read_readvariableop0savev2_adam_dense_0_kernel_v_read_readvariableop.savev2_adam_dense_0_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop0savev2_adam_tdist_0_kernel_v_read_readvariableop.savev2_adam_tdist_0_bias_v_read_readvariableop0savev2_adam_tdist_1_kernel_v_read_readvariableop.savev2_adam_tdist_1_bias_v_read_readvariableop0savev2_adam_tdist_2_kernel_v_read_readvariableop.savev2_adam_tdist_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�d:d:dd:d:dd:d:d:: : : : : :d:d:dd:d:	d�:�: : : : :	�d:d:dd:d:dd:d:d::d:d:dd:d:	d�:�:	�d:d:dd:d:dd:d:d::d:d:dd:d:	d�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:%!

_output_shapes
:	d�:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$  

_output_shapes

:d: !

_output_shapes
:d:$" 

_output_shapes

:dd: #

_output_shapes
:d:%$!

_output_shapes
:	d�:!%

_output_shapes	
:�:%&!

_output_shapes
:	�d: '

_output_shapes
:d:$( 

_output_shapes

:dd: )

_output_shapes
:d:$* 

_output_shapes

:dd: +

_output_shapes
:d:$, 

_output_shapes

:d: -

_output_shapes
::$. 

_output_shapes

:d: /

_output_shapes
:d:$0 

_output_shapes

:dd: 1

_output_shapes
:d:%2!

_output_shapes
:	d�:!3

_output_shapes	
:�:4

_output_shapes
: 
�
�
'__inference_output_layer_call_fn_569545

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_568469o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
I
-__inference_activation_4_layer_call_fn_569502

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_568434`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
d
H__inference_activation_4_layer_call_and_return_conditional_losses_568434

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������dZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
\
@__inference_mask_layer_call_and_return_conditional_losses_569423

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    p
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*4
_output_shapes"
 :������������������`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*0
_output_shapes
:������������������d
CastCastAny:output:0*

DstT0*

SrcT0
*0
_output_shapes
:������������������Y
IdentityIdentityCast:y:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
C__inference_dense_1_layer_call_and_return_conditional_losses_569603

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
B__inference_output_layer_call_and_return_conditional_losses_568469

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
P
$__inference_sum_layer_call_fn_569439
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_sum_layer_call_and_return_conditional_losses_568388a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:������������������:�������������������:Z V
0
_output_shapes
:������������������
"
_user_specified_name
inputs/0:_[
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
input;
serving_default_input:0������������������@
activation_60
StatefulPartitionedCall:0���������tensorflow/serving/predict:ͯ
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer_with_weights-6
layer-15
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	layer
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	(layer
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	5layer
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
�

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
�

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
�

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�beta_1
�beta_2

�decay
�learning_rate
	�iterNm�Om�\m�]m�jm�km�xm�ym�	�m�	�m�	�m�	�m�	�m�	�m�Nv�Ov�\v�]v�jv�kv�xv�yv�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
�0
�1
�2
�3
�4
�5
N6
O7
\8
]9
j10
k11
x12
y13"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
N6
O7
\8
]9
j10
k11
x12
y13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
&__inference_model_layer_call_fn_568514
&__inference_model_layer_call_fn_568923
&__inference_model_layer_call_fn_568956
&__inference_model_layer_call_fn_568776�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_layer_call_and_return_conditional_losses_569058
A__inference_model_layer_call_and_return_conditional_losses_569160
A__inference_model_layer_call_and_return_conditional_losses_568830
A__inference_model_layer_call_and_return_conditional_losses_568884�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_568075input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_tdist_0_layer_call_fn_569204
(__inference_tdist_0_layer_call_fn_569213�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_tdist_0_layer_call_and_return_conditional_losses_569234
C__inference_tdist_0_layer_call_and_return_conditional_losses_569255�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_activation_layer_call_fn_569260�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_activation_layer_call_and_return_conditional_losses_569265�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_tdist_1_layer_call_fn_569274
(__inference_tdist_1_layer_call_fn_569283�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_tdist_1_layer_call_and_return_conditional_losses_569304
C__inference_tdist_1_layer_call_and_return_conditional_losses_569325�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_activation_1_layer_call_fn_569330�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_1_layer_call_and_return_conditional_losses_569335�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_tdist_2_layer_call_fn_569344
(__inference_tdist_2_layer_call_fn_569353�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_tdist_2_layer_call_and_return_conditional_losses_569374
C__inference_tdist_2_layer_call_and_return_conditional_losses_569395�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�2�
%__inference_mask_layer_call_fn_569400
%__inference_mask_layer_call_fn_569405�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_mask_layer_call_and_return_conditional_losses_569414
@__inference_mask_layer_call_and_return_conditional_losses_569423�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_activation_2_layer_call_fn_569428�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_2_layer_call_and_return_conditional_losses_569433�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�2�
$__inference_sum_layer_call_fn_569439�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_sum_layer_call_and_return_conditional_losses_569449�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!:	�d2dense_0/kernel
:d2dense_0/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_0_layer_call_fn_569458�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_0_layer_call_and_return_conditional_losses_569468�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_activation_3_layer_call_fn_569473�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_3_layer_call_and_return_conditional_losses_569478�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 :dd2dense_1/kernel
:d2dense_1/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_1_layer_call_fn_569487�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_569497�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_activation_4_layer_call_fn_569502�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_4_layer_call_and_return_conditional_losses_569507�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 :dd2dense_2/kernel
:d2dense_2/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_2_layer_call_fn_569516�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_569526�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_activation_5_layer_call_fn_569531�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_5_layer_call_and_return_conditional_losses_569536�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:d2output/kernel
:2output/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_output_layer_call_fn_569545�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_output_layer_call_and_return_conditional_losses_569555�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_activation_6_layer_call_fn_569560�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_6_layer_call_and_return_conditional_losses_569565�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
 :d2tdist_0/kernel
:d2tdist_0/bias
 :dd2tdist_1/kernel
:d2tdist_1/bias
!:	d�2tdist_2/kernel
:�2tdist_2/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_signature_wrapper_569195input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
&__inference_dense_layer_call_fn_569574�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_layer_call_and_return_conditional_losses_569584�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_1_layer_call_fn_569593�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_569603�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_2_layer_call_fn_569612�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_569622�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
&:$	�d2Adam/dense_0/kernel/m
:d2Adam/dense_0/bias/m
%:#dd2Adam/dense_1/kernel/m
:d2Adam/dense_1/bias/m
%:#dd2Adam/dense_2/kernel/m
:d2Adam/dense_2/bias/m
$:"d2Adam/output/kernel/m
:2Adam/output/bias/m
%:#d2Adam/tdist_0/kernel/m
:d2Adam/tdist_0/bias/m
%:#dd2Adam/tdist_1/kernel/m
:d2Adam/tdist_1/bias/m
&:$	d�2Adam/tdist_2/kernel/m
 :�2Adam/tdist_2/bias/m
&:$	�d2Adam/dense_0/kernel/v
:d2Adam/dense_0/bias/v
%:#dd2Adam/dense_1/kernel/v
:d2Adam/dense_1/bias/v
%:#dd2Adam/dense_2/kernel/v
:d2Adam/dense_2/bias/v
$:"d2Adam/output/kernel/v
:2Adam/output/bias/v
%:#d2Adam/tdist_0/kernel/v
:d2Adam/tdist_0/bias/v
%:#dd2Adam/tdist_1/kernel/v
:d2Adam/tdist_1/bias/v
&:$	d�2Adam/tdist_2/kernel/v
 :�2Adam/tdist_2/bias/v�
!__inference__wrapped_model_568075�������NO\]jkxy;�8
1�.
,�)
input������������������
� ";�8
6
activation_6&�#
activation_6����������
H__inference_activation_1_layer_call_and_return_conditional_losses_569335r<�9
2�/
-�*
inputs������������������d
� "2�/
(�%
0������������������d
� �
-__inference_activation_1_layer_call_fn_569330e<�9
2�/
-�*
inputs������������������d
� "%�"������������������d�
H__inference_activation_2_layer_call_and_return_conditional_losses_569433t=�:
3�0
.�+
inputs�������������������
� "3�0
)�&
0�������������������
� �
-__inference_activation_2_layer_call_fn_569428g=�:
3�0
.�+
inputs�������������������
� "&�#��������������������
H__inference_activation_3_layer_call_and_return_conditional_losses_569478X/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� |
-__inference_activation_3_layer_call_fn_569473K/�,
%�"
 �
inputs���������d
� "����������d�
H__inference_activation_4_layer_call_and_return_conditional_losses_569507X/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� |
-__inference_activation_4_layer_call_fn_569502K/�,
%�"
 �
inputs���������d
� "����������d�
H__inference_activation_5_layer_call_and_return_conditional_losses_569536X/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� |
-__inference_activation_5_layer_call_fn_569531K/�,
%�"
 �
inputs���������d
� "����������d�
H__inference_activation_6_layer_call_and_return_conditional_losses_569565X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
-__inference_activation_6_layer_call_fn_569560K/�,
%�"
 �
inputs���������
� "�����������
F__inference_activation_layer_call_and_return_conditional_losses_569265r<�9
2�/
-�*
inputs������������������d
� "2�/
(�%
0������������������d
� �
+__inference_activation_layer_call_fn_569260e<�9
2�/
-�*
inputs������������������d
� "%�"������������������d�
C__inference_dense_0_layer_call_and_return_conditional_losses_569468]NO0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� |
(__inference_dense_0_layer_call_fn_569458PNO0�-
&�#
!�
inputs����������
� "����������d�
C__inference_dense_1_layer_call_and_return_conditional_losses_569497\\]/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� �
C__inference_dense_1_layer_call_and_return_conditional_losses_569603^��/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� {
(__inference_dense_1_layer_call_fn_569487O\]/�,
%�"
 �
inputs���������d
� "����������d}
(__inference_dense_1_layer_call_fn_569593Q��/�,
%�"
 �
inputs���������d
� "����������d�
C__inference_dense_2_layer_call_and_return_conditional_losses_569526\jk/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� �
C__inference_dense_2_layer_call_and_return_conditional_losses_569622_��/�,
%�"
 �
inputs���������d
� "&�#
�
0����������
� {
(__inference_dense_2_layer_call_fn_569516Ojk/�,
%�"
 �
inputs���������d
� "����������d~
(__inference_dense_2_layer_call_fn_569612R��/�,
%�"
 �
inputs���������d
� "������������
A__inference_dense_layer_call_and_return_conditional_losses_569584^��/�,
%�"
 �
inputs���������
� "%�"
�
0���������d
� {
&__inference_dense_layer_call_fn_569574Q��/�,
%�"
 �
inputs���������
� "����������d�
@__inference_mask_layer_call_and_return_conditional_losses_569414vD�A
:�7
-�*
inputs������������������

 
p 
� ".�+
$�!
0������������������
� �
@__inference_mask_layer_call_and_return_conditional_losses_569423vD�A
:�7
-�*
inputs������������������

 
p
� ".�+
$�!
0������������������
� �
%__inference_mask_layer_call_fn_569400iD�A
:�7
-�*
inputs������������������

 
p 
� "!��������������������
%__inference_mask_layer_call_fn_569405iD�A
:�7
-�*
inputs������������������

 
p
� "!��������������������
A__inference_model_layer_call_and_return_conditional_losses_568830�������NO\]jkxyC�@
9�6
,�)
input������������������
p 

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_568884�������NO\]jkxyC�@
9�6
,�)
input������������������
p

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_569058�������NO\]jkxyD�A
:�7
-�*
inputs������������������
p 

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_569160�������NO\]jkxyD�A
:�7
-�*
inputs������������������
p

 
� "%�"
�
0���������
� �
&__inference_model_layer_call_fn_568514u������NO\]jkxyC�@
9�6
,�)
input������������������
p 

 
� "�����������
&__inference_model_layer_call_fn_568776u������NO\]jkxyC�@
9�6
,�)
input������������������
p

 
� "�����������
&__inference_model_layer_call_fn_568923v������NO\]jkxyD�A
:�7
-�*
inputs������������������
p 

 
� "�����������
&__inference_model_layer_call_fn_568956v������NO\]jkxyD�A
:�7
-�*
inputs������������������
p

 
� "�����������
B__inference_output_layer_call_and_return_conditional_losses_569555\xy/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� z
'__inference_output_layer_call_fn_569545Oxy/�,
%�"
 �
inputs���������d
� "�����������
$__inference_signature_wrapper_569195�������NO\]jkxyD�A
� 
:�7
5
input,�)
input������������������";�8
6
activation_6&�#
activation_6����������
?__inference_sum_layer_call_and_return_conditional_losses_569449�q�n
g�d
b�_
+�(
inputs/0������������������
0�-
inputs/1�������������������
� "&�#
�
0����������
� �
$__inference_sum_layer_call_fn_569439�q�n
g�d
b�_
+�(
inputs/0������������������
0�-
inputs/1�������������������
� "������������
C__inference_tdist_0_layer_call_and_return_conditional_losses_569234���D�A
:�7
-�*
inputs������������������
p 

 
� "2�/
(�%
0������������������d
� �
C__inference_tdist_0_layer_call_and_return_conditional_losses_569255���D�A
:�7
-�*
inputs������������������
p

 
� "2�/
(�%
0������������������d
� �
(__inference_tdist_0_layer_call_fn_569204s��D�A
:�7
-�*
inputs������������������
p 

 
� "%�"������������������d�
(__inference_tdist_0_layer_call_fn_569213s��D�A
:�7
-�*
inputs������������������
p

 
� "%�"������������������d�
C__inference_tdist_1_layer_call_and_return_conditional_losses_569304���D�A
:�7
-�*
inputs������������������d
p 

 
� "2�/
(�%
0������������������d
� �
C__inference_tdist_1_layer_call_and_return_conditional_losses_569325���D�A
:�7
-�*
inputs������������������d
p

 
� "2�/
(�%
0������������������d
� �
(__inference_tdist_1_layer_call_fn_569274s��D�A
:�7
-�*
inputs������������������d
p 

 
� "%�"������������������d�
(__inference_tdist_1_layer_call_fn_569283s��D�A
:�7
-�*
inputs������������������d
p

 
� "%�"������������������d�
C__inference_tdist_2_layer_call_and_return_conditional_losses_569374���D�A
:�7
-�*
inputs������������������d
p 

 
� "3�0
)�&
0�������������������
� �
C__inference_tdist_2_layer_call_and_return_conditional_losses_569395���D�A
:�7
-�*
inputs������������������d
p

 
� "3�0
)�&
0�������������������
� �
(__inference_tdist_2_layer_call_fn_569344t��D�A
:�7
-�*
inputs������������������d
p 

 
� "&�#��������������������
(__inference_tdist_2_layer_call_fn_569353t��D�A
:�7
-�*
inputs������������������d
p

 
� "&�#�������������������