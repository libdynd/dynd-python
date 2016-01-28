find_package( PythonInterp REQUIRED)

function( postprocess_cython postprocess_script postprocess_target pyx_target c_target )
    get_target_property(_generated_files ${pyx_target} generated_files)

    # Add the command to run the postprocessing script on the generated C/C++ files.
    add_custom_target( ${postprocess_target}
                       COMMAND ${PYTHON_EXECUTABLE} ${postprocess_script} ${_generated_files}
                       COMMENT "Postprocessing generated C/C++ files from target ${pyx_target}."
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

    add_dependencies( ${postprocess_target} ${pyx_target} )
    add_dependencies( ${c_target} ${postprocess_target})
endfunction()
