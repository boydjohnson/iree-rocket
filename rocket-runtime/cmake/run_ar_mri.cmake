# Runs `${AR} -M < ${MRI}`: add_custom_command has no portable stdin redirect.
execute_process(
  COMMAND "${AR}" -M
  INPUT_FILE "${MRI}"
  RESULT_VARIABLE _result
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "${AR} -M < ${MRI} failed: ${_result}")
endif()
