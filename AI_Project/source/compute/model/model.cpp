//void mnist_model_save(mnist_model & model, const std::string & fname) {
//  printf("%s: saving model to '%s'\n", __func__, fname.c_str());
//
//  struct ggml_context * ggml_ctx;
//  {
//    struct ggml_init_params params = {
//      /*.mem_size   =*/ 100 * 1024*1024,
//      /*.mem_buffer =*/ NULL,
//      /*.no_alloc   =*/ false,
//  };
//    ggml_ctx = ggml_init(params);
//  }
//
//  gguf_context * gguf_ctx = gguf_init_empty();
//  gguf_set_val_str(gguf_ctx, "general.architecture", model.arch.c_str());
//
//  std::vector<struct ggml_tensor *> weights;
//  if (model.arch == "mnist-fc") {
//    weights = {model.fc1_weight, model.fc1_bias, model.fc2_weight, model.fc2_bias};
//  } else if (model.arch == "mnist-cnn") {
//    weights = {model.conv1_kernel, model.conv1_bias, model.conv2_kernel, model.conv2_bias, model.dense_weight, model.dense_bias};
//  } else {
//    GGML_ASSERT(false);
//  }
//  for (struct ggml_tensor * t : weights) {
//    struct ggml_tensor * copy = ggml_dup_tensor(ggml_ctx, t);
//    ggml_set_name(copy, t->name);
//    ggml_backend_tensor_get(t, copy->data, 0, ggml_nbytes(t));
//    gguf_add_tensor(gguf_ctx, copy);
//  }
//  gguf_write_to_file(gguf_ctx, fname.c_str(), false);
//
//  ggml_free(ggml_ctx);
//  gguf_free(gguf_ctx);
//}
