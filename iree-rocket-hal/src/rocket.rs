#[allow(non_upper_case_globals, non_camel_case_types)]
pub mod api;
pub mod builders;
pub mod conv;
pub mod debug;
pub mod device;
pub mod executable_format;
pub mod fc;
pub mod lut_tables;
pub mod regcmd;

// One module per hardware op, split out of what used to be a single
// `regcmd` grab-bag. `regcmd` itself now holds only what all of them
// share (see its module doc comment).
//
// `conv` above and `mesa_conv` here are the same op from two independent
// sources -- decoded vendor `.rknn` captures and Mesa's `rkt_regcmd.c`
// respectively. `pooling`, `activation`, and `elementwise` currently
// carry only the Mesa-derived lineage; their capture-derived builders
// belong in these same modules as they are written.
pub mod activation;
pub mod elementwise;
pub mod mesa_conv;
pub mod pooling;
#[allow(non_upper_case_globals, non_camel_case_types)]
pub mod registers;
pub mod tensor_layout;
